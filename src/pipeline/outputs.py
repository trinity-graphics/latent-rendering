from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum, auto
from typing import TypeGuard

import torch

from pipeline.mode import PipelineMode


class Reduction(Enum):
    """How repeated scalar logs within one step collapse at commit()."""
    SINGLE = auto()   # exactly one value expected; logging twice is a bug
    MEAN = auto()     # one or more values expected; averaged

def is_pipeline_output(obj: object) -> TypeGuard[OutputDataContainer]:
    if not is_dataclass(obj):
        return False
    return all(
        "meta" in f.metadata and isinstance(f.metadata["meta"], OutputMeta)
        for f in fields(obj)
    )

@dataclass
class OutputData:
    """
    Pipeline output types.
    - Image is used for storing RGB image tensors: [1, Res, Res, 3]
    - Latent is used for storing latent tensors: [1, Res/8, Res/8, 16] (for SD3.5)
    - Scalar is used for storing individual floats e.g. losses, timings.
    """

    data: torch.Tensor


@dataclass
class Image(OutputData):
    gt: torch.Tensor | None = None


@dataclass
class Latent(OutputData):
    gt: torch.Tensor | None = None
    decode_gt: torch.Tensor | None = None


@dataclass
class Scalar(OutputData):
    pass


@dataclass
class OutputMeta:
    dtype: type[OutputData]
    pretty_name: str
    description: str
    log_decoded: bool = False
    key_template: str | None = None
    has_ground_truth: bool = False
    active_modes: PipelineMode | None = None
    requires: str | None = None
    reduction: Reduction = Reduction.SINGLE


_OUTPUT_TYPES: dict[str, type[OutputData]] = {
    cls.__name__: cls for cls in (Image, Latent, Scalar)
}

def snapshot(
    dtype: type[OutputData] | str,
    t: torch.Tensor,
    gt: torch.Tensor | None = None,
    decode_gt: torch.Tensor | None = None,
) -> OutputData:
    """
    Creates a CPU copy of the tensor for logging later.
    """
    if isinstance(dtype, str):
        dtype = _OUTPUT_TYPES[dtype]

    data = t.detach().clone().cpu()
    field_names = {f.name for f in fields(dtype)}

    kwargs: dict = {"data": data}
    if "gt" in field_names and gt is not None:
        kwargs["gt"] = gt.detach().clone().cpu()
    if "decode_gt" in field_names and decode_gt is not None:
        kwargs["decode_gt"] = decode_gt.detach().clone().cpu()

    return dtype(**kwargs)



@dataclass
class OutputDataContainer:
    pass


#
##
### Primary pipeline outputs.
##
#
@dataclass
class PipelineOutputs(OutputDataContainer):

    final: Latent | None = field(
        default=None,
        metadata={
            "meta": OutputMeta(
                dtype=Latent,
                pretty_name="Final Latent",
                description="Final latent after all pipeline steps.",
                log_decoded=True,
                active_modes=PipelineMode.ALL,
                has_ground_truth=True,
            )
        },
    )

    processed: Latent | None = field(
        default=None,
        metadata={
            "meta": OutputMeta(
                dtype=Latent,
                pretty_name="Processed Render",
                description="Raw rendered latent before ambient addition.",
                log_decoded=True,
                active_modes=PipelineMode.SCENE | PipelineMode.EVAL,
                has_ground_truth=True,
            )
        },
    )
    ambient: Latent | None = field(
        default=None,
        metadata={
            "meta": OutputMeta(
                dtype=Latent,
                pretty_name="Ambient",
                description="Ambient lighting component.",
                active_modes=PipelineMode.SCENE | PipelineMode.EVAL,
            )
        },
    )
    post_ambient: Latent | None = field(
        default=None,
        metadata={
            "meta": OutputMeta(
                dtype=Latent,
                pretty_name="Post Ambient",
                description="Latent after ambient addition.",
                log_decoded=True,
                active_modes=PipelineMode.SCENE | PipelineMode.EVAL,
                has_ground_truth=True,
            )
        },
    )
    refined: Latent | None = field(
        default=None,
        metadata={
            "meta": OutputMeta(
                dtype=Latent,
                pretty_name="Refined",
                description="Latent after convolutional refinement.",
                log_decoded=True,
                active_modes=PipelineMode.REFINER | PipelineMode.EVAL,
                has_ground_truth=True,
                requires="use_refiner",
            )
        },
    )
    pre_residual: Latent | None = field(
        default=None,
        metadata={
            "meta": OutputMeta(
                dtype=Latent,
                pretty_name="Pre Residual",
                description="Residual of single pre-refiner layer.",
                active_modes=PipelineMode.REFINER,
                requires="pre_edge_conv",
            )
        },
    )
    pre_sample: Latent | None = field(
        default=None,
        metadata={
            "meta": OutputMeta(
                dtype=Latent,
                pretty_name="Pre Sample",
                description="Latent after the single pre-refiner layer.",
                log_decoded=True,
                active_modes=PipelineMode.REFINER | PipelineMode.EVAL,
                has_ground_truth=True,
                requires="pre_edge_conv",
            )
        },
    )
    layer_samples: list[Latent] | None = field(
        default=None,
        metadata={
            "meta": OutputMeta(
                dtype=Latent,
                pretty_name="Layer {i} Sample",
                description="Per-layer sample from the refiner at layer {i}.",
                key_template="layer{i}_sample",
                active_modes=PipelineMode.REFINER | PipelineMode.EVAL,
                has_ground_truth=True,
                requires="per_layer_residuals",
            )
        },
    )
    layer_residuals: list[Latent] | None = field(
        default=None,
        metadata={
            "meta": OutputMeta(
                dtype=Latent,
                pretty_name="Layer {i} Residual",
                description="Per-layer residual from the refiner at layer {i}.",
                key_template="layer{i}_residual",
                active_modes=PipelineMode.REFINER,
                requires="per_layer_residuals",
            )
        },
    )
    loss_1: Scalar | None = field(
        default=None,
        metadata={
            "meta": OutputMeta(
                dtype=Scalar,
                pretty_name="Render Loss",
                description="Scene rendering loss, averaged across views.",
                active_modes=PipelineMode.SCENE,
                reduction=Reduction.MEAN,
            )
        },
    )
    loss_2: Scalar | None = field(
        default=None,
        metadata={
            "meta": OutputMeta(
                dtype=Scalar,
                pretty_name="Refiner Loss",
                description="Refiner network loss, averaged across views.",
                active_modes=PipelineMode.REFINER,
                reduction=Reduction.MEAN,
            )
        },
    )

    @property
    def total_loss(self) -> torch.Tensor | None:
        losses = [l.data for l in (self.loss_1, self.loss_2) if l is not None]
        return sum(losses) if losses else None


#
##
### Pipeline outputs when using the time_forward function.
##
#
@dataclass
class PipelineTimingOutputs(OutputDataContainer):
    time_rgb_render: Scalar | None = field(
        default=None,
        metadata={"meta": OutputMeta(
            dtype=Scalar,
            pretty_name="GT Render Time (ms)",
            description="DrTimer wall-clock for rm.rgb.render(), ms.",
            active_modes=PipelineMode.EVAL,
        )},
    )
    time_rgb_encode: Scalar | None = field(
        default=None,
        metadata={"meta": OutputMeta(
            dtype=Scalar,
            pretty_name="GT Encode Time (ms)",
            description="bench.Timer wall-clock for vae.encode(), ms.",
            active_modes=PipelineMode.EVAL,
        )},
    )
    time_latent_render: Scalar | None = field(
        default=None,
        metadata={"meta": OutputMeta(
            dtype=Scalar,
            pretty_name="Latent Render Time (ms)",
            description="DrTimer wall-clock for rm.latent.render(), ms.",
            active_modes=PipelineMode.EVAL,
        )},
    )
    time_latent_refine: Scalar | None = field(
        default=None,
        metadata={"meta": OutputMeta(
            dtype=Scalar,
            pretty_name="Latent Refine Time (ms)",
            description="bench.Timer wall-clock for the conv refiner, ms.",
            active_modes=PipelineMode.EVAL,
            requires="use_refiner",
        )},
    )
    time_latent_decode: Scalar | None = field(
        default=None,
        metadata={"meta": OutputMeta(
            dtype=Scalar,
            pretty_name="Latent Decode Time (ms)",
            description="bench.Timer wall-clock for vae.decode(), ms.",
            active_modes=PipelineMode.EVAL,
        )},
    )


#
##
### Test-time outputs.
##
#
@dataclass
class TestOutputs(OutputDataContainer):
    gt_rgb: Image | None = field(
        default=None,
        metadata={
            "meta": OutputMeta(
                dtype=Image,
                pretty_name="GT RGB",
                description="Ground truth RGB render.",
                active_modes=PipelineMode.EVAL,
                has_ground_truth=True,
            )
        },
    )
    gt_latent: Latent | None = field(
        default=None,
        metadata={
            "meta": OutputMeta(
                dtype=Latent,
                pretty_name="GT Latent",
                description="Ground truth encoded latent.",
                active_modes=PipelineMode.EVAL,
                has_ground_truth=True,
            )
        },
    )
    
    best_rgb: Image | None = field(
        default=None,
        metadata={"meta": OutputMeta(
            dtype=Image,
            pretty_name="Best RGB Reference",
            description="High-SPP reference image used as GT.",
            active_modes=PipelineMode.EVAL,
            has_ground_truth=False,
        )},
    )
    best_latent: Latent | None = field(
        default=None,
        metadata={"meta": OutputMeta(
            dtype=Latent,
            pretty_name="Best Latent Reference",
            description="Encoded high-SPP reference latent used as GT.",
            active_modes=PipelineMode.EVAL,
            has_ground_truth=False,
            log_decoded=False,
        )},
    )   