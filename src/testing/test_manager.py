import logging
from collections.abc import Callable
from pathlib import Path

import drjit as dr
import pandas as pd
import torch
import torch.utils.benchmark as bench
from tqdm.auto import tqdm

from latents.mi_vae import MiVae
from logger import Logger
from pipeline import (
    Image,
    Latent,
    LatentRenderingPipeline,
    PipelineMode,
    PipelineTimingOutputs,
    Scalar,
    TestOutputs,
    snapshot,
)
from testing.dr_timer import DrTimer
from training.train_manager import ReferenceImages, TrainingManager


def update_dict_keys(d: dict, prefix: str = "", suffix: str = ""):
    if len(prefix) > 0:
        prefix = prefix + "_"
    if len(suffix) > 0:
        suffix = "_" + suffix
    return {f"{prefix}{key}{suffix}": value for key, value in d.items()}



class TestingManager:
    def __init__(self, cfg: dict, log: Logger, vae: MiVae):
        self.log: Logger = log
        self.vae = vae

        self.out_dir: Path = Path("outputs") / cfg["out_dir"]

        # Resolve the scene + refiner checkpoints to load
        joint_ckpt = self.out_dir / "joint.pt"
        local_scene = self.out_dir / "scene.pt"
        external_scene = (
            Path("outputs") / cfg["scene_dir"] / "scene.pt" if cfg.get("scene_dir") else None
        )
        local_refiner = self.out_dir / "refiner.pt"

        if joint_ckpt.exists():
            self.scene_ckpt = joint_ckpt
            self.refiner_ckpt = joint_ckpt
        else:
            self.scene_ckpt = local_scene if local_scene.exists() else external_scene
            self.refiner_ckpt = local_refiner

        # Pipeline timing (disabled by default)
        self.time_forward = cfg.get("time_forward", False)

        self.TRACKED_METRICS = ["mse", "psnr", "ssim", "lpips"]

        self.VID_FMT = "mp4"
        self.VID_FPS = 60

        self.experiments = []

    def register_experiment(
        self,
        exp_name: str,
        scene_modify_fn: Callable,
        iterations: int,
    ):
        self.experiments.append(
            {
                "exp_name": exp_name,
                "exp_fn": scene_modify_fn,
                "exp_its": iterations,
            }
        )

    def get_aggregated_metrics(
        self, csv_path: Path, ignore_frame=-1,
    ):
        # Defines a custom aggregation function.  Ignores the row with it == ignore_frame.
        df = pd.read_csv(csv_path)
        return df[df["it"] != ignore_frame].drop(columns="it").mean().to_dict()


    def prepare_best_reference(self, pipeline, best_spp: int = 3969):
        rm = pipeline.rm
        spp_before = rm.rgb.spp
        rm.rgb.set_spp(best_spp)
        with torch.no_grad(), dr.suspend_grad():
            best_rgb = rm.rgb.render(denoiser=True)
            best_latent = self.vae.encode(best_rgb)
        rm.rgb.set_spp(spp_before)
        return best_rgb, best_latent 


    def run_tests(
        self,
        pipeline: LatentRenderingPipeline
    ):
        rm = pipeline.rm
        nets = pipeline.nets
        pipeline.eval()
        self.vae.vae.eval()
        with torch.no_grad(), dr.suspend_grad():
            for exp in self.experiments:
                pipeline.mode = PipelineMode.EVAL
                self.log.begin_experiment(exp["exp_name"])

                exp_path = self.log._experiment_dir()
                exp_images = self.log.experiment_images_dir()

                print(f'\nRunning experiment "{exp["exp_name"]}"...')

                rm.reset_renderers()
                rm.load_scene_checkpoint(self.scene_ckpt)
                rm.set_optimizable_parameters(TrainingManager.SCENE_OPTIMIZATION_TARGETS)
                nets.load_checkpoint(self.refiner_ckpt)

                rm.set_debug_out_dir(exp_images)
                nets.set_debug_out_dir(exp_images)

                for it in tqdm(range(exp["exp_its"]), dynamic_ncols=True):
                    self.log.set_step(it)
                    self.log.set_is_main_view(True)

                    exp["exp_fn"](it, exp["exp_its"])

                    best_rgb, best_latent = self.prepare_best_reference(pipeline, best_spp=3969)

                    # RGB ground truth (+ optional render timing)
                    if self.time_forward:
                        alt_render_ms = DrTimer(lambda: rm.rgb.render(seed=it), warmup=1).timeit(1)
                    gt_rgb = rm.rgb.render(seed=it, denoiser=False)

                    if self.time_forward:
                        alt_encode_ms = bench.Timer(
                            stmt="vae.encode(img)",
                            globals={"vae": self.vae, "img": gt_rgb},
                        ).timeit(1).mean * 1e3
                    gt_latent = self.vae.encode(gt_rgb)

                    refs = ReferenceImages(
                        img=best_rgb,
                        latent_mean=None,
                        latent_logvar=None,
                        latent_processed=best_latent,
                        latent_sample=best_latent,
                    )

                    # Full pipeline forward — render + ambient + refinement
                    out = pipeline(seed=it, refs=refs, snap=True)

                    test_out = TestOutputs(
                        best_rgb=snapshot(Image, best_rgb),
                        best_latent=snapshot(Latent, best_latent),
                        gt_rgb=snapshot(Image, gt_rgb, gt=best_rgb),
                        gt_latent=snapshot(Latent, gt_latent, gt=best_latent),
                    )

                    self.log.log_outputs(out, mode=PipelineMode.EVAL)
                    self.log.log_outputs(test_out, mode=PipelineMode.EVAL)

                    # Latent pipeline timings (separate pass) — only when enabled
                    if self.time_forward:
                        latent_timing = pipeline.time_forward(seed=it)
                        timing_out = PipelineTimingOutputs(
                            time_rgb_render=Scalar(data=torch.tensor(alt_render_ms)),
                            time_rgb_encode=Scalar(data=torch.tensor(alt_encode_ms)),
                            time_latent_render=latent_timing.time_latent_render,
                            time_latent_refine=latent_timing.time_latent_refine,
                            time_latent_decode=latent_timing.time_latent_decode,
                        )
                        self.log.log_outputs(timing_out, mode=PipelineMode.EVAL)

                    self.log.commit(mode=PipelineMode.EVAL, force=it == exp["exp_its"] - 1)

                    del gt_rgb, gt_latent, best_rgb, best_latent, refs, out, test_out
                    if self.time_forward:
                        del timing_out

                # Video export — one per schema debug output
                self.log.make_experiment_videos(exp["exp_name"], exp_path)
                self.log.end_experiment()



    def benchmark(
        self,
        pipeline: LatentRenderingPipeline,
        gt_rgb_best,
        gt_latent_best,
        n: int = 10,
        rgb_spp=32,
        latent_spp=2048,
    ):
        import torch.utils.benchmark as bench

        from utils.img_utils import compute_image_metrics

        from .dr_timer import DrTimer

        logging.info("Benchmarking pipeline performance...")

        pipeline.eval()
        self.vae.vae.eval()
        rm   = pipeline.rm
        nets = pipeline.nets

        spp_scenes = {
            # 256: "outputs/benchmark/Lamp-256spp",
            # 1024: "outputs/Lamp-1024-Best",
            # 2048: "outputs/benchmark/Lamp-2048spp"
            256: "outputs/CBox-1024-Best",
            1024: "outputs/CBox-1024-Best",
            2048: "outputs/CBox-1024-Best",
        }

        scene_ckpt = Path(spp_scenes[latent_spp]) / "scene.pt"
        refiner_ckpt = Path(spp_scenes[latent_spp]) / "refiner.pt"

        rm.reset_renderers()
        rm.load_scene_checkpoint(scene_ckpt)
        rm.set_optimizable_parameters(TrainingManager.SCENE_OPTIMIZATION_TARGETS)
        nets.load_checkpoint(refiner_ckpt)

        seed = 0
        rm.rgb.set_spp(rgb_spp)
        rm.latent.set_spp(latent_spp)

        rm.set_debug_out_dir(Path("./benchmark"))
        nets.set_debug_out_dir(Path("./benchmark"))

        metrics = {}

        with torch.no_grad(), dr.suspend_grad():
            # 1. GT RGB render — drjit kernel timing
            def _rgb_render(seed):
                rm.rgb.render(denoiser=False, seed=seed)

            rgb_m = DrTimer(_rgb_render, warmup=1, step_kwargs=["seed"]).timeit(n)
            gt_rgb = rm.rgb.render(seed=seed, denoiser=False)
            metrics["rgb_render_rgb_space"], _ = compute_image_metrics(gt_rgb_best, gt_rgb)
            nets.save_debug_image(gt_rgb, "gt_rgb", debug=rgb_spp, save_exr=True)

            # 2. GT RGB → latent encode — torch.utils.benchmark
            enc_m = bench.Timer(
                stmt="vae.encode(img)",
                globals={"vae": self.vae, "img": gt_rgb},
            ).timeit(n).mean
            gt_latent = self.vae.encode(gt_rgb)
            metrics["rgb_render_latent_space"], _ = compute_image_metrics(gt_latent_best, gt_latent, is_latent=True)
            nets.save_debug_image(gt_latent, "gt_latent", debug=rgb_spp, save_exr=True)

            # 3. Latent render — drjit kernel timing
            def _latent_render(seed):
                rm.latent.render(seed=seed)
            latent_m = DrTimer(_latent_render, warmup=1, step_kwargs=["seed"]).timeit(n)
            latent, _dbg = rm.latent.render(seed=seed)

            # 4. Latent refinement — torch.utils.benchmark
            # AOVs pre-computed outside the timer: it's a drjit op, not network time
            aovs = (
                rm.aov.render(rm.rgb.scene, rm.latent.params, seed=seed)
                if rm.aov is not None else None
            )

            def _refine(lat, aovs):
                if nets.is_refiner_enabled():
                    lat, _ = nets.refine_latent_conv(lat, aovs)
                return lat

            ref_m = bench.Timer(
                stmt="_refine(lat, aovs)",
                globals={"_refine": _refine, "lat": latent, "aovs": aovs},
            ).timeit(n).mean
            rendered_latent = _refine(latent, aovs)
            metrics["latent_render_latent_space"], _ = compute_image_metrics(gt_latent_best, rendered_latent, is_latent=True)

            nets.save_debug_image(rendered_latent, "rendered_latent", debug=latent_spp, save_exr=True)

            dec_m = bench.Timer(
                stmt="vae.decode(lat)",
                globals={"vae": self.vae, "lat": gt_latent},
            ).timeit(n).mean
            decoded_rendered_latent = self.vae.decode(rendered_latent)
            metrics["latent_render_rgb_space"], _ = compute_image_metrics(gt_rgb_best, decoded_rendered_latent)

            nets.save_debug_image(decoded_rendered_latent, "decoded_rendered_latent", debug=latent_spp, save_exr=True)

            del gt_rgb, gt_latent, latent

        def log_benchmark_metrics(stage):
            m = metrics[stage]
            logging.info(f"     MSE*100:        {m['mse'] * 100:.3f}")
            logging.info(f"     MSE*100 (lin):  {m['mse_lin'] * 100:.3f}")
            logging.info(f"     SSIM:           {m['ssim']:.3f}")
            if "lpips" in m:
                logging.info(f"     LPIPS:          {m['lpips']:.3f}")

        logging.info(f"--- Benchmark (n={n}, RGB spp={rgb_spp}, Latent spp={latent_spp}) ---")
        logging.info(f"  GT RGB render:      {rgb_m:.3f} ms")
        log_benchmark_metrics("rgb_render_rgb_space")
        
        logging.info(f"  GT RGB encode:      {enc_m * 1e3:.3f} ms")   # s → ms
        log_benchmark_metrics("rgb_render_latent_space")

        logging.info(f"  Latent render:      {latent_m:.3f} ms")
        
        logging.info(f"  Latent refinement:  {ref_m * 1e3:.3f} ms")   # s → ms
        log_benchmark_metrics("latent_render_latent_space")
        
        logging.info(f"  Latent decode:      {dec_m * 1e3:.3f} ms")   # s → ms
        log_benchmark_metrics("latent_render_rgb_space")

        logging.info("")
        logging.info("")
        logging.info("")
        logging.info("")
        logging.info("")