from __future__ import annotations

import datetime
import logging
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from random import uniform
from typing import TYPE_CHECKING

import drjit as dr
import mitsuba as mi
import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

from pipeline import LatentRenderingPipeline, PipelineMode
from utils import io_utils

# from dataset.dataset import LatentDataset
from utils.losses import (
    EarlyStopper,
    gaussian_regularizer,
    hard_regularization,
    huber,
    kldiv,
    lpipsLoss,
    mse,
    power_regularizer,
    ssimLoss,
)

if TYPE_CHECKING:
    from latents.latent_processor import LatentProcessor
    from latents.mi_vae import MiVae
    from logger import Logger
    from networks import NetworkManager
    from renderers import RendererManager


def debug_print_iterator(it, debug_name):
    logging.debug(f"\n### [DEBUG] {debug_name} ###")
    for k, v in it:
        logging.debug(f"{k}: {v}")
    logging.debug("")


def no_batch(batch):
    return batch[0]


@dataclass
class ReferenceImages:
    img: mi.TensorXf
    latent_mean: mi.TensorXf
    latent_logvar: mi.TensorXf
    latent_processed: mi.TensorXf
    latent_sample: mi.TensorXf | None = None
    latent_denoised: mi.TensorXf | None = None
    latent_blurred: mi.TensorXf | None = None


class TrainingManager:

    SCENE_OPTIMIZATION_TARGETS = [
        ".miss_color",
        ".tex_scale.value",
        ".tex_offset.value",
        ".diffuse_reflectance.value",
        ".reflectance.value",
        ".specular_reflectance.value",
        "specular_transmittance.value",
        ".albedo.value",
        ".sigma_t.value",
        ".scale.value",
        ".ambient.value",
        ".neg_ambient.value",
        ".occ_ambient.value",
        ".neg_occ_ambient.value",
        ".radiance.value",
        ".color0.value",
        ".color1.value",
        # ".alpha_u.value",
        # ".alpha_v.value",
        # ".alpha.value"
    ]

    # ━━━━━━━━━━━━━━━━━━
    # MARK: Init
    # ━━━━━━━━━━━━━━━━━━
    def __init__(self, cfg: dict, log: Logger, vae: MiVae):
        self.out_dir = Path("outputs") / cfg["out_dir"]
        self.vae = vae

        local_scene = self.out_dir / "scene.pt"
        external_scene = (
            Path("outputs") / cfg["scene_dir"] / "scene.pt" if cfg.get("scene_dir") else None
        )
        self.scene_ckpt = local_scene if local_scene.exists() else external_scene

        # Prepare debug parameters
        self.debug_images = cfg["debug_images"]
        self.debug_image_steps = cfg["debug_image_steps"]
        self.debug_metrics = cfg["debug_metrics"]
        self.debug_metric_steps = cfg["debug_metric_steps"]
        self.debug_out_warned = []

        # Prepare optimization logging
        self.log = log
        # self.log_steps = cfg["log_steps"]
        # self.latent_metrics = {}
        # # self.latent_channel_metrics = {}

        # Prepare training data source parameters
        self.dataset_training = cfg["dataset_training"]
        self.dataset_name = cfg["dataset"]
        self.random_pos_training = cfg["random_pos"]

        # Prepare training iteration parameters
        self.num_its = cfg["num_its"]
        self.multi_stage = cfg["multi_stage"]
        self.stage_split = cfg["stage_split"]
        self.dataset_early_stop = cfg["dataset_early_stop"]

        self.latent_spp = cfg["samples_per_pixel_latents"]
        self.latent_spp_training = cfg.get("samples_per_pixel_latents_training", None)

        # Prepare losses
        loss_fns = {"huber": huber, "mse": mse}
        self.loss_weight_1 = cfg["loss1_weight"]
        self.loss_fn_1 = loss_fns[cfg["loss1_type"]]
        self.loss_fn_2 = loss_fns[cfg["loss2_type"]]
        self.SSIM_1 = cfg["SSIM_loss_1"]
        self.SSIM_2 = cfg["SSIM_loss_2"]
        self.LPIPS_1 = cfg["LPIPS_loss_1"]
        self.LPIPS_2 = cfg["LPIPS_loss_2"]

        self.regularizers = {
            "Gaussian": gaussian_regularizer,
            "KL": kldiv,
            "NLL": self.vae.nll,
            "power": power_regularizer,
            "hard": hard_regularization,
        }
        self.KL_loss_1 = cfg["KL_loss_1"]
        self.Gaussian_loss_1 = cfg["Gaussian_loss_1"]
        self.NLL_loss_1 = cfg["NLL_loss_1"]
        self.hard_loss_1 = cfg["hard_loss_1"]
        self.power_loss_1 = cfg["power_loss_1"]
        self.KL_loss_2 = cfg["KL_loss_2"]
        self.Gaussian_loss_2 = cfg["Gaussian_loss_2"]
        self.NLL_loss_2 = cfg["NLL_loss_2"]
        self.hard_loss_2 = cfg["hard_loss_2"]
        self.power_loss_2 = cfg["power_loss_2"]

        # Prepare loss targets
        self.train_on_dist_samples = cfg["train_on_dist_samples"]
        self.sample_every_it = cfg["sample_every_it"]
        self.loss1_denoised_latent = cfg["loss1_denoised_latent"]
        self.loss1_blurred = cfg["loss1_blurred"]
        self.loss2_target = cfg["loss_target"]

        # Prepare lr scheduler for scene opt
        self.lr_min = cfg["lr_min"]
        self.lr_max = cfg["lr_max"]
        self.warmup = cfg["warmup"]
        self.c_len = cfg["cycle_len"]
        self.decay = cfg["decay"]
        self.v_fac = cfg["vanishing_factor"]

        # Prepare reference move mode
        self.ref_view_mode = cfg["move_mode"]

        self.set_to_view = {
            "camera": self.set_to_camera_view,
            "light": self.set_to_light_view,
            "object": self.set_to_object_view,
        }[self.ref_view_mode]

        # Optional ground-truth image override
        self.gt_image = None
        gt_path = cfg.get("gt_image", None)
        if gt_path:
            from utils import img_utils

            bmp = mi.Bitmap(gt_path).convert(
                mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32, srgb_gamma=True
            )
            img = img_utils.mi2pt(mi.TensorXf(bmp)).to(self.vae.vae.device)
            res = cfg["image_resolution"]
            if img.shape[-2:] != (res, res):
                raise ValueError(
                    f"gt_image `{gt_path}` has resolution {tuple(img.shape[-2:])}, "
                    f"expected ({res}, {res})."
                )
            self.gt_image = img
            print(f"[gt_image] Using `{gt_path}` as the ground truth.")

        # Early stopper
        self.early_stopper = EarlyStopper(
            cfg["stopper_patience"],
            cfg["stopper_min_delta"],
            cfg["stopper_stag_eps"],
            None,
        )

        self.quick_scene_optim = cfg["quick_scene_optim"]

        # Refiner training data
        self.train_refiner_saved = cfg["train_refiner_saved"]
        self.augment_channel_shuffle = cfg.get("augment_channel_shuffle", False)

        self.saved_cfg = cfg

        self.TRAINING_FUNCTIONS = {f.__name__: f for f in [
            self.get_render_loss,
            self.get_refiner_loss,
            self.get_residual_losses,
        ]}

    # ━━━━━━━━━━━━━━━━━━
    # MARK: Helpers
    # ━━━━━━━━━━━━━━━━━━
    def prepare_scene_optimization(self, renderers: RendererManager, param_substrings: Iterable[str]):
        renderers.set_optimizable_parameters(param_substrings)
        self.scene_optimizer = torch.optim.Adam(
            renderers.scene_opt_params.values(), lr=self.lr_max
        )
        self.scene_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=self.scene_optimizer, T_0=self.c_len, eta_min=self.lr_min
        )
        self.early_stopper.set_opt(self.scene_optimizer)
        debug_print_iterator(renderers.scene_opt_params.items(), "Optimization Parameters")

    def prepare_network_optimization(self, networks: NetworkManager, lr: float):
        if networks.is_refiner_enabled():
            self.refiner_optimizer = torch.optim.Adam(
                lr=lr, params=networks.miniRefiner.parameters()
            )

    def prepare_dataset(
        self, cfg: dict, latent_shape: tuple[int], latent_render_shape: tuple[int]
    ):
        DS_DIR = "dataset"
        if self.dataset_training:
            raise NotImplementedError(
                "LatentDataset class needs rewrite to match new BCHW impl!"
            )

            # self.dataset = LatentDataset(
            #     cfg=cfg,
            #     data_dir=f"{DS_DIR}/{self.dataset_name}",
            #     metadata_file=f"{DS_DIR}/{self.dataset_name}_metadata.csv",
            #     saved_cfg_file=f"{DS_DIR}/dataset_cfg.yaml",
            #     latent_shape=latent_shape,
            #     processed_latent_shape=latent_render_shape,
            # )
            # # TODO: separate out train and val with random_split and implement proper validation
            # self.dataloader = iter(
            #     DataLoader(
            #         self.dataset_name,
            #         batch_size=1,
            #         num_workers=0,
            #         collate_fn=no_batch,
            #         pin_memory=True,
            #     )
            # )

    def render_references(
        self, rm: RendererManager, data: dict = None, debug: bool = False
    ):
        with torch.no_grad():
            # Reference image
            if self.gt_image is not None:
                img_ref = self.gt_image
            elif data is not None:
                img_ref = mi.TensorXf(data["image"])
                img_ref = dr.clip(img_ref, 0, 1)
                img_ref = dr.power(img_ref, 1.0 / 2.2)
            else:
                img_ref = rm.rgb.render(debug)

            # Latent distribution (mean + variance)
            if data is not None and data["latent_params"] is not None:
                lat_params = mi.TensorXf(data["latent_params"])
            else:
                lat_params = self.vae.encode(img_ref, return_parameters=True)
            lat_ref_mean, lat_ref_logvar = self.vae.preprocess_parameters(lat_params)
            del lat_params

            # Latent distribution sample, OR scaled mean
            if self.train_on_dist_samples:
                lat_sample = self.vae.sample_latent_distribution(
                    lat_ref_mean, lat_ref_logvar
                )
            else:
                lat_sample = self.vae.scale_mean(lat_ref_mean)
                lat_ref_mean, lat_ref_logvar = None, None

            # Denoised latent
            lat_ref_denoised = None
            if self.loss1_denoised_latent:
                if data is not None and data["mean_denoised"] is not None:
                    lat_ref_denoised = mi.TensorXf(data["mean_denoised"])
                else:
                    lat_ref_denoised = rm.latent.latent_processor.denoise_latent(lat_sample)

            # Processed latent
            if data is not None and data["mean_processed"] is not None:
                lat_ref_processed = mi.TensorXf(data["mean_processed"])
            else:
                lat_ref_processed = (
                    rm.latent.latent_processor.preprocess_latent_to_render(lat_sample)
                )

            # Blurred latent
            if self.loss1_blurred:
                if self.loss1_denoised_latent:
                    lat_ref_blurred = rm.latent.latent_processor.blur_latent(lat_ref_denoised)
                else:
                    lat_ref_blurred = rm.latent.latent_processor.blur_latent(lat_sample)
            else:
                lat_ref_blurred = None

            return ReferenceImages(
                img=img_ref,
                latent_mean=lat_ref_mean,
                latent_logvar=lat_ref_logvar,
                latent_sample=lat_sample,
                latent_processed=lat_ref_processed,
                latent_denoised=lat_ref_denoised,
                latent_blurred=lat_ref_blurred,
            )

    def set_to_camera_view(self, rm: RendererManager, view_idx: int, data: dict = None):
        if data is not None:
            az, el = data["az"], data["el"]
        elif self.random_pos_training:
            az, el = [uniform(-(i / 2), i / 2) for i in [rm.x_range, rm.y_range]]
        else:
            az, el = rm.view_positions[view_idx]

        rm.rotate_sensors(az, el)

    def set_to_light_view(self, rm: RendererManager, view_idx: int, data: dict = None):
        if data is not None:
            print("WE DO NOT HAVE DATA YET!")

        offset = rm.view_positions[view_idx]

        rm.move_light(*offset)

    def set_to_object_view(self, rm: RendererManager, view_idx: int, data: dict = None):
        if data is not None:
            print("WE DO NOT HAVE DATA YET!")

        offset = rm.view_positions[view_idx]

        rm.move_object(*offset)

    def get_references(self, rm: RendererManager, view_idx: int, data: dict = None):
        if self.dataset_training:
            refs = self.render_references(rm, data=data)
        elif self.random_pos_training:
            refs = self.render_references(rm)
        else:
            refs = rm.ref_views[view_idx]

        return refs

    def get_cached_latents(self, rm: RendererManager, view_idx: int, seed: int = 0):
        if self.train_refiner_saved:
            renders = rm.rendered_views[view_idx]
            idx = seed % len(renders)
            return renders[idx], rm.rendered_views_debug[view_idx][idx]
        with torch.no_grad(), dr.suspend_grad():
            return rm.latent.render(seed, opt_params=rm.get_scaled_opt_params())

    def prepare_training_iterations(self):
        # Account for dataset size in iteration count.
        if self.dataset_training:
            ds_size = len(self.dataloader)
            if self.num_its > ds_size:
                print(
                    f"WARNING: Wanted {self.num_its} iterations, but there are only {ds_size} images in the dataset. Will stop early."
                )
            else:
                if not self.dataset_early_stop:
                    self.num_its = ds_size
                else:
                    print(
                        f"`dataset_early_stop` enabled, will stop after {self.num_its}/{ds_size} dataset images."
                    )

        if self.multi_stage:
            split = int(self.num_its * self.stage_split)
            self.scene_its = range(split)
            self.refiner_its = range(split, self.num_its)
            self.joint_its = None
        else:
            self.scene_its = None
            self.refiner_its = None
            self.joint_its = range(self.num_its)

    # ━━━━━━━━━━━━━━━━━━
    # MARK: I/O
    # ━━━━━━━━━━━━━━━━━━
    def save_config(self, config_file_path):
        cfg = {k: v for k, v in self.saved_cfg.items() if k != "__path__"}
        with open(self.out_dir / "run_config.yaml", "w") as f:
            f.write(f"# Merged from {config_file_path}\n")
            yaml.safe_dump(cfg, f, sort_keys=False)

    def make_picklable(self, obj: object):
        if dr.is_array_v(obj):
            return obj.numpy()
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu()
        if isinstance(obj, np.ndarray):
            return obj
        if isinstance(obj, (list, tuple)):
            return type(obj)(self.make_picklable(x) for x in obj)
        if isinstance(obj, dict):
            return {str(k): self.make_picklable(v) for k, v in obj.items()}
        return obj

    def save_checkpoint(
        self, rm: RendererManager, nets: NetworkManager, mode: PipelineMode
    ):
        ckpt = {
            "created": datetime.datetime.now().isoformat(),
            "torch": torch.__version__,
            "config": self.saved_cfg,
        }

        if PipelineMode.SCENE in mode:
            ckpt.update(
                {
                    "scene_params": self.make_picklable(rm.scene_opt_params),
                    "latent_data": self.make_picklable(
                        asdict(rm.latent.latent_processor.latent_data)
                    ),
                }
            )
            self.save_optimized_scene(rm)

        if PipelineMode.REFINER in mode:
            ckpt.update(
                {
                    "refiner": nets.miniRefiner.state_dict()
                    if nets.is_refiner_enabled()
                    else None,
                    "refiner_chs": rm.latent.latent_processor.LATENT_CH,
                }
            )

        torch.save(ckpt, str(self.out_dir / f"{mode.name.lower()}.pt"))

    def save_optimized_scene(self, rm: RendererManager):
        """Write a human-readable scene XML with the optimized values baked in."""
        cfg = self.saved_cfg
        src_xml = io_utils.get_latent_xml_fname(
            Path(cfg["scene_file"]), io_utils.get_impl_details(cfg)
        )
        if not src_xml.exists():
            logging.warning(
                f"Optimized scene XML not written: source `{src_xml}` not found."
            )
            return
        opt_values = {
            k: p.detach().cpu().flatten().tolist()
            for k, p in rm.scene_opt_params.items()
        }
        unmatched = io_utils.write_optimized_scene(
            src_xml, self.out_dir / "scene.xml", opt_values
        )
        if unmatched:
            logging.warning(
                f"Optimized scene XML written, but {len(unmatched)} optimized "
                f"parameter(s) had no matching <latent> element: {unmatched}"
            )

    # ━━━━━━━━━━━━━━━━━━
    # MARK: Losses
    # ━━━━━━━━━━━━━━━━━━
    def get_residual_losses(
        self, lp: LatentProcessor, refs: ReferenceImages, residuals: dict
    ):
        res_loss = 0
        if "layer_samples" in residuals:
            for ls in residuals["layer_samples"]:
                res_loss += huber(refs.latent_sample, ls)
        if "pre_sample" in residuals:
            res_loss += huber(refs.latent_sample, residuals["pre_sample"])

        return res_loss

    def get_refiner_loss(self, refs: ReferenceImages, latent: mi.TensorXf):
        if self.loss2_target == "latent":
            if self.train_on_dist_samples and self.sample_every_it:
                true_2 = self.vae.sample_latent_distribution(
                    refs.latent_mean, refs.latent_logvar
                )
            else:
                true_2 = refs.latent_sample
            test_2 = latent
        elif self.loss2_target == "image":
            true_2 = refs.img
            test_2 = self.vae.decode(latent)
        else:
            raise ValueError("Invalid loss target specified in config file.")

        loss_2 = self.loss_fn_2(true_2, test_2)
        if self.SSIM_2:
            loss_2 += ssimLoss(true_2, test_2)
        for regname, regularizer in self.regularizers.items():
            if getattr(self, f"{regname}_loss_2"):
                loss_2 += regularizer(test_2)
        if self.LPIPS_2:
            true_2 = refs.img
            test_2 = self.vae.decode(latent)
            loss_2 += lpipsLoss(true_2, test_2)
        return loss_2

    def get_render_loss(self, rm: RendererManager, refs: ReferenceImages, latent):
        if self.train_on_dist_samples and self.sample_every_it:
            true_1 = self.vae.sample_latent_distribution(
                refs.latent_mean, refs.latent_logvar
            )
        else:
            true_1 = refs.latent_sample
        test_1 = latent

        if self.loss1_denoised_latent:
            true_1 = refs.latent_denoised  # Denoised mean
            test_1 = test_1
        elif self.loss1_blurred:
            true_1 = refs.latent_blurred  # Blurred mean
            test_1 = rm.latent.latent_processor.blur_latent(test_1)

        loss_1 = self.loss_fn_1(true_1, test_1)
        if self.SSIM_1:
            loss_1 += ssimLoss(true_1, test_1)
        for regname, regularizer in self.regularizers.items():
            if getattr(self, f"{regname}_loss_1"):
                loss_1 += regularizer(test_1)
        if self.LPIPS_1:
            true_1 = refs.img
            test_1 = self.vae.decode(latent)
            loss_1 += lpipsLoss(true_1, test_1)
        return loss_1

    # ━━━━━━━━━━━━━━━━━━
    # MARK: Main
    # ━━━━━━━━━━━━━━━━━━
    def train(self, pipeline: LatentRenderingPipeline, mode=PipelineMode.JOINT):
        pipeline.mode = mode
        rm = pipeline.rm
        nets = pipeline.nets
        logger = self.log
        kill = False
        cached_render = None

        train_its = getattr(self, f"{mode.name.lower()}_its")
        if train_its is None:
            print(f"No {mode.name.lower()} optimization scheduled, skipping...")
            return
        desc = f"{mode.name.title()} Optimization"
        prog_bar = tqdm(train_its, dynamic_ncols=True, desc=desc)

        for epoch in prog_bar:
            logger.set_step(epoch)
            log_due = logger.should_snapshot()

            loss_1s_per_view = []
            loss_2s_per_view = []

            if PipelineMode.SCENE in mode:
                self.scene_optimizer.zero_grad()

            for view_idx in range(rm.num_views):
                is_main = view_idx == rm.main_view_idx
                logger.set_is_main_view(is_main)

                data = next(self.dataloader) if self.dataset_training else None
                self.set_to_view(rm, view_idx, data=data)
                refs = self.get_references(rm, view_idx, data=data)

                if mode == PipelineMode.REFINER:
                    cached_render = self.get_cached_latents(rm, view_idx, epoch)

                    if self.augment_channel_shuffle:
                        n_ch = cached_render[0].shape[1]
                        perm = torch.randperm(n_ch, device="cpu")
                        cached_render = cached_render[0][:, perm, :, :], cached_render[1]
                        refs.latent_sample = refs.latent_sample[:, perm, :, :]

                out = pipeline(
                    seed=epoch,
                    refs=refs,
                    cached_render=cached_render,
                    train_fns=self.TRAINING_FUNCTIONS,
                    snap=log_due and is_main,
                )
                out.total_loss.backward()

                loss_1s_per_view.append(out.loss_1.data.item() if out.loss_1 else 0.0)
                loss_2s_per_view.append(out.loss_2.data.item() if out.loss_2 else 0.0)

                if log_due:
                    logger.log_outputs(out, mode=mode)

                del out, data, refs

            loss_1_mean = sum(loss_1s_per_view) / rm.num_views
            loss_2_mean = sum(loss_2s_per_view) / rm.num_views
            logger.commit(mode=mode, force=epoch == self.num_its - 1)

            # Update latent rendering parameters
            if PipelineMode.SCENE in mode:
                kill = self.early_stopper.update(loss_1_mean, self.scene_optimizer)
                self.scene_optimizer.step()

                # Parameter clipping
                with torch.no_grad():
                    idx_reflect_terms = (epoch * 3) / len(train_its)
                    for k, p in rm.scene_opt_params.items():
                        if all([ref_term not in k for ref_term in rm.AMBIENT_TERMS]):
                            clip_bound = 6.0
                        elif idx_reflect_terms < 1 and not self.quick_scene_optim:
                            clip_bound = 1e-4 if "occlusion" in k else 0.1
                        else:
                            continue
                        if rm.negative_latent_rendering:
                            p.clamp_(-clip_bound, clip_bound)
                        else:
                            p.clamp_(0, clip_bound)

                if kill:
                    if idx_reflect_terms < 1:
                        print(
                            f"{mode.name.lower()} optimization started diverging, process getting moved to unrestricted ambient terms at step {epoch}"
                        )
                        epoch = int((len(train_its) / 3) + 1)
                        kill = False
                        self.early_stopper.counter = 0
                    self.scene_optimizer = self.early_stopper.opt_min

                self.scene_scheduler.step()

            # Update refiner network parameters
            if PipelineMode.REFINER in mode:
                if nets.is_refiner_enabled():
                    self.refiner_optimizer.step()
                    self.refiner_optimizer.zero_grad()

            if kill:
                print(
                    f"{mode.name.lower()} optimization started diverging, process killed at step {epoch}"
                )
                break

        #
        #   Optimization Complete
        #
        self.scene_optimizer = self.early_stopper.opt_min
        if hasattr(rm, "rendered_views"):
            del rm.rendered_views
        if hasattr(rm, "rendered_views_debug"):
            del rm.rendered_views_debug
        
        # Save a checkpoint
        self.save_checkpoint(rm, nets, mode)
        
        # Flush
        dr.flush_malloc_cache()
        dr.sync_thread()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def run(self, pipeline, refiner_only_training: bool):
        if refiner_only_training:
            pipeline.rm.load_scene_checkpoint(self.scene_ckpt)

        pipeline.rm.get_reference_views(get_refs_fn=self.render_references, out_dir=self.out_dir)

        self.prepare_scene_optimization(pipeline.rm, self.SCENE_OPTIMIZATION_TARGETS)
        self.prepare_network_optimization(pipeline.nets, lr=self.saved_cfg["lr"])
        self.prepare_training_iterations()

        if not refiner_only_training:
            if self.latent_spp_training is not None:
                pipeline.rm.latent.set_spp(self.latent_spp_training)
            try:
                self.train(pipeline, PipelineMode.SCENE)
            finally:
                pipeline.rm.latent.set_spp(self.latent_spp)

        pipeline.rm.get_rendered_views(
            out_dir=self.out_dir,
            refiner_renders_per_view=self.saved_cfg["refiner_train_renders"],
        )

        if self.saved_cfg["use_refiner"]:
            self.train(pipeline, PipelineMode.REFINER)
            self.train(pipeline, PipelineMode.JOINT)