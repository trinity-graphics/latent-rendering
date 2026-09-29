import dataclasses
import json
import logging
import sys
from collections.abc import Iterable
from pathlib import Path

import torch

import wandb
from logger.metrics import LoggerMetrics, _MetricsEncoder
from pipeline.mode import PipelineMode
from pipeline.outputs import (
    Image,
    Latent,
    OutputDataContainer,
    OutputMeta,
    Reduction,
    Scalar,
)
from utils import img_utils


####################################################################################################
#   Logger: A class that combines logging capabilities for JSON, WandB, and the logging library.   #
####################################################################################################
class Logger:
    ROOT_OUTPUT_DIR = "outputs"
    OUTPUT_SUBDIRS = ["views", "debug"]
    VID_FMT = "mp4"
    VID_FPS = 60
    PAD_DIGITS = 4

    def __init__(
        self,
        args,
        cfg: dict,
        metrics: LoggerMetrics,
        vae,
    ):
        self.vae = vae
        self.experiment_name: str | None = None

        # Set up local outputs
        self.out_dir = Path(self.ROOT_OUTPUT_DIR) / cfg["out_dir"]
        self.make_output_directories(self.OUTPUT_SUBDIRS)

        self.image_out_dir = self.out_dir / "debug"

        # Set up logging intervals
        self.debug_images = cfg["debug_images"]
        self.debug_image_steps = cfg["debug_image_steps"]
        self.debug_metrics = cfg["debug_metrics"]
        self.debug_metric_steps = cfg["debug_metric_steps"]
        self.debug_out_warned = []

        self.log_steps = cfg["log_steps"]

        # Set up WandB
        name = Path(args.config).stem
        if args.train_only:
            name += "_training"
        if args.test_only:
            name += "_test"
        self.wandb = wandb.init(
            entity=cfg["team"],
            project=cfg["project"],
            name=name,
            config=cfg,
            mode="online" if (cfg["log"] and args.test_log) else "disabled",
        )

        #
        #   Set up logging
        #
        self._log = logging.getLogger()
        self._log.setLevel(logging.DEBUG)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler = logging.FileHandler(self.out_dir / "debug.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self._log.addHandler(file_handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self._log.addHandler(console_handler)

        # _step is internal tracking of the training or testing step.
        self._step: int = -1
        self._is_main_view: bool = False
        self._in_context = False
        # _data is the current step's data.
        self.metrics: LoggerMetrics = metrics
        # _data_batch is the accumulated data of every `self.log_steps` steps, for I/O & upload efficiency.
        self.metrics_batch: list[dict] = []

        self._initialized_jsonl_paths: set = set()
        self._scalar_accum: dict[str, list[float]] = {}

    def make_output_directories(self, dirs: Iterable[str]):
        self.out_dir.mkdir(exist_ok=True, parents=True)
        for p in [self.out_dir / d for d in dirs]:
            p.mkdir(exist_ok=True)

    @property
    def jsonl_path(self) -> Path:
        if self.experiment_name is None:
            return self.out_dir / "metrics.jsonl"
        return self._experiment_dir() / f"{self.experiment_name}.jsonl"

    #######################
    #   Step management   #
    #######################
    def set_step(self, step: int) -> None:
        self._step = step

    def set_is_main_view(self, is_main_view) -> None:
        self._is_main_view = is_main_view

    def _in_experiment(self) -> bool:
        return self.experiment_name is not None

    def _metrics_step(self) -> bool:
        return self._in_experiment() or (
            self.debug_metrics and self._step % self.debug_metric_steps == 0
        )

    def _images_step(self) -> bool:
        return self._in_experiment() or (
            self.debug_images and self._step % self.debug_image_steps == 0
        )

    def metrics_due(self) -> bool:
        return self._metrics_step() and (self._in_experiment() or self._is_main_view)

    def images_due(self) -> bool:
        return self._images_step() and (self._in_experiment() or self._is_main_view)

    def should_snapshot(self) -> bool:
        return self._metrics_step() or self._images_step()

    def _flush_due(self, force: bool = False) -> bool:
        return force or self._step % self.log_steps == 0

    ##############################
    #   Main logging functions   #
    ##############################

    def log_scalar(self, key: str, value: float) -> None:
        meta, _ = self.metrics[key]
        if meta.dtype is not Scalar:
            raise RuntimeError(f"'{key}' is not a scalar!")
        self._scalar_accum.setdefault(key, []).append(value)

    def log_image(self, key: str, image, gt=None, decode_gt=None) -> None:
        meta, _ = self.metrics[key]

        if meta.dtype is not Image and meta.dtype is not Latent:
            raise TypeError(f"'{key}' is not an image or latent!")
        if image is None:
            raise ValueError("Cannot log None-valued image.")

        out_dir = (
            self.experiment_images_dir()
            if self._in_experiment()
            else self.image_out_dir
        )

        # Save image locally
        if self.images_due():
            prefix = "latent_" if meta.dtype is Latent else ""
            out_path = out_dir / f"{prefix}{key}_{self._step:0{self.PAD_DIGITS}d}"

            if meta.dtype is Latent:
                img_utils.save_latent_png(image, out_path)
                img_utils.save_latent_exr(image, out_path)
            else:
                img_utils.save_image_png(image, out_path)
                img_utils.save_image_exr(image, out_path)

        # Calculate metrics against ground truth
        if self.metrics_due() and meta.has_ground_truth:
            if gt is None:
                raise ValueError(
                    f"'{key}' expects a ground truth but none was provided."
                )
            if image.shape != gt.shape:
                raise ValueError(
                    f"Shape mismatch between true {gt.shape} and test {image.shape}"
                )
            else:
                metrics, error_map = img_utils.compute_image_metrics(
                    true=gt, test=image, is_latent=meta.dtype is Latent
                )
                self.metrics[key] = metrics

                if error_map is not None and self.images_due():
                    error_map_path = (
                        out_dir / f"flip_{key}_{self._step:0{self.PAD_DIGITS}d}"
                    )
                    img_utils.save_image_png(error_map, error_map_path)

        # Decode and recursively log
        if (
            meta.dtype is Latent
            and meta.log_decoded
            and (self.images_due() or self.metrics_due())
        ):
            decoded_key = f"decoded_{key}"

            with torch.no_grad():
                decoded = self.vae.decode(image).cpu()
            self.log_image(decoded_key, decoded, gt=decode_gt)

    def _log_one(self, key: str, item) -> None:
        if isinstance(item, Scalar):
            self.log_scalar(key, item.data.item())
        elif isinstance(item, (Image, Latent)):
            self.log_image(key, item.data, gt=getattr(item, "gt", None), decode_gt=getattr(item, "decode_gt", None))
        else:
            raise TypeError(f"Unhandled OutputData type: {type(item).__name__}")

    def _field_in_schema(self, f, meta: OutputMeta) -> bool:
        if meta.key_template is not None:
            return meta.key_template.format(i=0) in self.metrics._schema
        return f.name in self.metrics._schema

    def log_outputs(self, out: OutputDataContainer, mode: PipelineMode | None = None) -> None:
        log_images = self.images_due() or self.metrics_due()
        for f in dataclasses.fields(out):
            meta: OutputMeta = f.metadata.get("meta")
            if meta is None:
                raise ValueError(
                    f"Field '{f.name}' in {type(out).__name__} has no OutputMeta. "
                    "Every field in an OutputDataContainer must define OutputMeta."
                )
            if not self._field_in_schema(f, meta):
                continue
            if mode is not None and meta.active_modes is not None and not (meta.active_modes & mode):
                continue
            value = getattr(out, f.name)
            if meta.dtype is Scalar:
                if value is None:
                    raise ValueError(
                        f"Active scalar '{f.name}' is None in {type(out).__name__}."
                    )
                self._log_one(f.name, value)
            else:
                if not log_images:
                    continue
                if value is None:
                    raise ValueError(
                        f"Active output '{f.name}' is None in {type(out).__name__} on its logging step."
                    )
                if isinstance(value, list):
                    for i, item in enumerate(value):
                        self._log_one(meta.key_template.format(i=i), item)
                else:
                    self._log_one(f.name, value)

    def log_videos(self, videos: dict) -> None:
        if self.wandb:
            self.wandb.log(videos, step=self._step, commit=False)

    def make_experiment_videos(self, exp_name: str, exp_path: Path) -> None:
        videos = {}
        images_dir = self.experiment_images_dir()
        for key, meta in self.metrics._schema.items():
            if meta.dtype is Scalar:
                continue
            # Outputs that are never logged at test time have no frames to encode.
            if meta.active_modes is not None and PipelineMode.EVAL not in meta.active_modes:
                continue
            prefix = "latent_" if meta.dtype is Latent else ""
            in_imgs = images_dir / f"{prefix}{key}_%0{self.PAD_DIGITS}d.png"
            out_file = exp_path / f"{prefix}{key}"
            img_utils.write_mp4(str(in_imgs), str(out_file), self.VID_FPS)
            videos[f"{exp_name}_{prefix}{key}"] = wandb.Video(
                out_file.with_suffix(f".{self.VID_FMT}"),
                fps=self.VID_FPS,
                format=self.VID_FMT,
            )
        self.log_videos(videos)

    ####################
    #   Commit logic   #
    ####################
    def commit(self, mode: PipelineMode | None = None, force: bool = False):
        if not self._metrics_step():
            self._scalar_accum.clear()
            self.metrics.reset()
            return

        for key, values in self._scalar_accum.items():
            meta = self.metrics._schema[key]
            if meta.reduction is Reduction.SINGLE and len(values) != 1:
                raise RuntimeError(
                    f"'{key}' is SINGLE but received {len(values)} values at step {self._step}."
                )
            self.metrics[key] = (
                values[0]
                if meta.reduction is Reduction.SINGLE
                else sum(values) / len(values)
            )
        self._scalar_accum.clear()

        incomplete = self.incomplete_entries(mode)
        if incomplete:
            raise RuntimeError(
                f"Logger.commit() was called with unlogged active entries: {incomplete}. "
                "Either log to them or deactivate them before committing."
            )

        self.metrics_batch.append(self._build_row())
        self.metrics.reset()

        if self._flush_due(force):
            self._flush(force)

    def _build_row(self) -> dict:
        row = {"step": self._step}
        for key in self.metrics.all_keys():
            meta, data = self.metrics[key]
            row[key] = data if meta.dtype is Scalar else dict(data) if data else None
        return row

    def _flush(self, force: bool = False) -> None:
        if not self.metrics_batch:
            raise RuntimeError("Flush attempted with no data.")
        self._flush_jsonl()
        self._flush_wandb(force)
        self._log_metrics()
        self.metrics_batch.clear()

    def _flush_wandb(self, force: bool = False) -> None:
        if not self.wandb:
            return
        for row in self.metrics_batch:
            payload = {}
            for name, value in row.items():
                if name == "step":
                    continue
                meta = self.metrics._schema[name]
                if meta.dtype is Scalar:
                    payload[name] = value
                elif value:
                    for metric, v in value.items():
                        payload[f"{name}/{metric}"] = v
            self.wandb.log(
                payload,
                step=row["step"],
                commit=(force or row is self.metrics_batch[-1]),
            )

    def _flush_jsonl(self) -> None:
        if not self.jsonl_path or not self.metrics_batch:
            return
        path = self.jsonl_path
        mode = "w" if path not in self._initialized_jsonl_paths else "a"
        self._initialized_jsonl_paths.add(path)
        with path.open(mode) as f:
            for row in self.metrics_batch:
                f.write(json.dumps(row, cls=_MetricsEncoder) + "\n")

    def _log_metrics(self) -> None:
        if not self.metrics_due():
            return
        self._log.debug(f"\n### Metrics — Step {self._step} ###")
        for key in self.metrics.all_keys():
            meta, data = self.metrics[key]
            self._log.debug(f"  [{meta.pretty_name}]")
            if meta.dtype is Scalar:
                self._log.debug(f"    {data}")
            elif data:
                for metric, value in data.items():
                    self._log.debug(f"    {metric}: {value}")

    # Experiment management
    def begin_experiment(self, name: str) -> None:
        self.experiment_name = name

    def end_experiment(self) -> None:
        self.experiment_name = None

    # Misc helpers
    def _experiment_dir(self) -> Path:
        if self.experiment_name:
            d = self.out_dir / self.experiment_name
            d.mkdir(exist_ok=True)
            return d
        return self.image_out_dir

    def experiment_images_dir(self) -> Path:
        d = self._experiment_dir() / "images"
        d.mkdir(exist_ok=True)
        return d

    def incomplete_entries(self, mode: PipelineMode | None = None) -> list[str]:
        result = []
        for k in self.metrics.all_keys():
            meta = self.metrics._schema[k]
            if mode is not None and meta.active_modes is not None and not (meta.active_modes & mode):
                continue
            if meta.dtype is not Scalar and not meta.has_ground_truth:
                continue
            if k not in self.metrics.logged_entries:
                result.append(k)
        return result

    def finish(self) -> None:
        if self.wandb:
            self.wandb.finish()
