from argparse import Namespace

import torch

from args import parse_args
from config import load_config, parse_config_overrides
from dtype import DType
from latents.mi_vae import MiVae

device = "cuda" if torch.cuda.is_available() else "cpu"
DS_DIR = "dataset"


def make_experiments(tm, renderers, cfg: dict):
    def lerp_move(target: str, it, max_its):
        t = it / max_its
        offsets = []
        for axis in ("x", "y", "z"):
            min_v = cfg.get(f"{target}_min_{axis}", 0.0) or 0.0
            max_v = cfg.get(f"{target}_max_{axis}", 0.0) or 0.0
            offset = min_v + t * (max_v - min_v)
            offsets.append(offset)
        return tuple(offsets)

    def move_camera(it, max_its):
        degrees = lerp_move("camera", it, max_its)
        renderers.rotate_sensors(degrees[0], degrees[1])
        return degrees

    def move_object(it, max_its):
        offset = lerp_move("object", it, max_its)
        renderers.move_object(*offset)
        return offset

    def move_light(it, max_its):
        offset_light_object = lerp_move("light_object", it, max_its)
        offset_light = lerp_move("light", it, max_its)

        renderers.move_light(*offset_light)
        renderers.move_light_object(*offset_light_object)
        return offset_light_object

    def move_all(it, max_its):
        offset_object = lerp_move("object", it, max_its)
        offset_light_object = lerp_move("light_object", it, max_its)
        offset_light = lerp_move("light", it, max_its)

        renderers.move_light(*offset_light)
        renderers.move_light_object(*offset_light_object)
        renderers.move_object(*offset_object)
        return offset_object

    # Latent spp grows as it^2, at the training view.
    def convergence_best(it, max_its):
        renderers.latent.set_spp(it * it)
        return (0, 0, 0)

    # Latent spp grows as it^2, at the final move_camera view.
    def convergence_novel_best(it, max_its):
        offset_camera = lerp_move("camera", max_its, max_its)
        renderers.rotate_sensors(offset_camera[0], offset_camera[1])
        renderers.latent.set_spp(it * it)
        return (0, 0, 0)

    experiments = {
        fn.__name__: fn
        for fn in (
            move_camera,
            move_object,
            move_light,
            move_all,
            convergence_best,
            convergence_novel_best,
        )
    }
    for name in cfg["experiments"]:
        if name not in experiments:
            raise ValueError(
                f"Unknown experiment `{name}`, must be one of {list(experiments)}."
            )
        tm.register_experiment(name, experiments[name], 60)


def set_latent_rendering_impl(device, cfg: dict, lat_ch: int):
    from mitsuba import set_variant, variants

    mi_device = "cuda" if device == "cuda" else "llvm"

    if cfg["custom_mitsuba"]:
        if any("latent" in v for v in variants()):
            if cfg["split_channels"]:  # Split channels into +/- components
                lat_ch *= 2
            if cfg["invert"]:  # Clone and invert channels
                lat_ch *= 2
            try:
                mi_colortype = f"latent{lat_ch}"
            except ImportError as e:
                print(
                    f"The custom latent Mitsuba does not have a latent variant supporting {lat_ch} channels."
                )
                print(e)
        else:
            print(
                "`custom_mitsuba` was enabled in the config, but "
                "Mitsuba was not compiled with a latent variant. "
                "Defaulting to spectral implementation, which may "
                "have worse performance and unexpected behavior."
            )
            mi_colortype = "spectral"
    else:
        mi_colortype = "spectral"

    set_variant(f"{mi_device}_ad_{mi_colortype}")


#
# MARK:     Main
#
def main(args: Namespace, cfg: dict):
    # torch.autograd.set_detect_anomaly(True)
    # dr.set_flag(dr.JitFlag.Debug, True)

    if args.train_only and args.test_only:
        print("train_only and test_only were both enabled, exiting.")
        exit()

    #
    ### Prepare pipeline and logging
    #
    pipeline = LatentRenderingPipeline(cfg, mi_vae, device)
    output_classes = [PipelineOutputs, TestOutputs]
    if cfg.get("time_forward", False):
        output_classes.append(PipelineTimingOutputs)
    log_metrics = LoggerMetrics(
        *output_classes,
        num_refiner_layers=pipeline.num_refiner_layers,
        exclude=pipeline.inactive_outputs,
    )
    log = Logger(args, cfg, log_metrics, mi_vae)

    #
    ### TRAINING
    #
    if args.test_only:
        print(
            "test_only enabled, skipping optimization and "
            f"loading values from {cfg['out_dir']}.\n"
        )
    else:
        trm = TrainingManager(cfg, log, mi_vae)
        trm.save_config(args.config)
        trm.run(pipeline, args.refiner_training)

    #
    ### TESTING
    #
    if args.train_only:
        print(
            "train_only enabled, skipping testing and "
            f"saving optimized values to {cfg['out_dir']}.\n"
        )
    else:
        tem = TestingManager(cfg, log, mi_vae)
        make_experiments(tem, pipeline.rm, cfg)
        # if args.benchmark:
        #     pipeline.rm.rgb.set_spp(3969)
        #     gt_rgb = pipeline.rm.rgb.render(denoiser=True)
        #     gt_lat = mi_vae.encode(gt_rgb)
        #     N = 10
        #     for spp in (
        #         #RGB, Latent
        #         (4, 256),
        #         (32, 2048),
        #         (2048, 2048),
        #     ):
        #         tem.benchmark(pipeline, gt_rgb, gt_lat, N, *spp)
        # else:

        tem.run_tests(pipeline)


#
# MARK:     Setup
#
if __name__ == "__main__":
    #
    #       Args
    #
    args, overrides = parse_args()

    #
    #       Config
    #
    cfg = load_config(args.config, print_loaded_configs=True)
    overrides = parse_config_overrides(cfg, overrides)
    cfg.update(overrides)

    #
    #       Type Handling
    #
    dtype = DType(cfg["dtype"])

    #
    #       VAE
    #
    mi_vae = MiVae(
        model_name=cfg["model_id"], dtype=dtype.torch, device=device
    )

    #
    #       Mitsuba Variant
    #
    set_latent_rendering_impl(
        device,
        cfg,
        mi_vae.latent_channels()
    )

    # log = Logger(cfg)

    # These imports must happen after variant setting to support mitsuba function calls.
    from logger import Logger, LoggerMetrics
    from pipeline import (
        LatentRenderingPipeline,
        PipelineOutputs,
        PipelineTimingOutputs,
        TestOutputs,
    )
    from testing import TestingManager
    from training import TrainingManager

    main(args, cfg)
