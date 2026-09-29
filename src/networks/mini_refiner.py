import math
from typing import Any

import torch
from torch import nn

try:
    from inplace_abn import InPlaceABN
except ImportError:
    InPlaceABN = None


# Helper modules/functions sourced from
# https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/segmentation_models_pytorch/base/modules.py
def get_norm_layer(
    norm: bool | str | dict[str, Any], out_channels: int
) -> nn.Module:
    supported_norms = ("inplace", "batchnorm", "identity", "layernorm", "instancenorm")

    # Step 1. Convert tot dict representation

    ## Check boolean
    if norm is True:
        norm_params = {"type": "batchnorm"}
    elif norm is False:
        norm_params = {"type": "identity"}

    ## Check string
    elif isinstance(norm, str):
        norm_str = norm.lower()
        if norm_str == "inplace":
            norm_params = {
                "type": "inplace",
                "activation": "leaky_relu",
                "activation_param": 0.0,
            }
        elif norm_str in supported_norms:
            norm_params = {"type": norm_str}
        else:
            raise ValueError(
                f"Unrecognized normalization type string provided: {norm}. Should be in "
                f"{supported_norms}"
            )

    ## Check dict
    elif isinstance(norm, dict):
        norm_params = norm

    else:
        raise ValueError(
            f"Invalid type for use_norm should either be a bool (batchnorm/identity), "
            f"a string in {supported_norms}, or a dict like {{'type': 'batchnorm', **kwargs}}"
        )

    # Step 2. Check if the dict is valid
    if "type" not in norm_params:
        raise ValueError(
            f"Malformed dictionary given in use_norm: {norm}. Should contain key 'type'."
        )
    if norm_params["type"] not in supported_norms:
        raise ValueError(
            f"Unrecognized normalization type string provided: {norm}. Should be in {supported_norms}"
        )
    if norm_params["type"] == "inplace" and InPlaceABN is None:
        raise RuntimeError(
            "In order to use `use_norm='inplace'` the inplace_abn package must be installed. Use:\n"
            "  $ pip install -U wheel setuptools\n"
            "  $ pip install inplace_abn --no-build-isolation\n"
            "Also see: https://github.com/mapillary/inplace_abn"
        )

    # Step 3. Initialize the norm layer
    norm_type = norm_params["type"]
    norm_kwargs = {k: v for k, v in norm_params.items() if k != "type"}

    if norm_type == "inplace":
        norm_out = InPlaceABN(out_channels, **norm_kwargs)
    elif norm_type == "batchnorm":
        norm_out = nn.BatchNorm2d(out_channels, **norm_kwargs)
    elif norm_type == "identity":
        norm_out = nn.Identity()
    elif norm_type == "layernorm":
        norm_out = nn.LayerNorm(out_channels, **norm_kwargs)
    elif norm_type == "instancenorm":
        norm_out = nn.InstanceNorm2d(out_channels, **norm_kwargs)
    else:
        raise ValueError(f"Unrecognized normalization type: {norm_type}")

    return norm_out



def get_activation_layer(act: str | dict[str, Any]):
    supported_acts = ("relu", "lrelu", "gelu", "silu", "mish", "tanh")

    # Step 1. Convert tot dict representation

    ## Check string
    if isinstance(act, str):
        act_str = act.lower()

        if act_str in supported_acts:
            act_params = {"type": act_str}
        else:
            raise ValueError(
                f"Unrecognized normalization type string provided: {act}. Should be in "
                f"{supported_acts}"
            )
        
    ## Check dict 
    elif isinstance(act, dict):
        act_params = act

    else:
        raise ValueError(
            f"Invalid type for use_norm should either be a bool (batchnorm/identity), "
            f"a string in {supported_acts}, or a dict like {{'type': 'batchnorm', **kwargs}}"
        )
    
    # Step 2. Check if the dict is valid
    if "type" not in act_params:
        raise ValueError(
            f"Malformed dictionary given in use_norm: {act}. Should contain key 'type'."
        )
    if act_params["type"] not in supported_acts:
        raise ValueError(
            f"Unrecognized normalization type string provided: {act}. Should be in {supported_acts}"
        )

    # Step 3. Initialize the activation layer
    act_type = act_params["type"]
    act_kwargs = {k: v for k, v in act_params.items() if k != "type"}
    if act_type == "relu":
        act_out = nn.ReLU(**act_kwargs)
    elif act_type == "lrelu":
        act_out = nn.LeakyReLU(**act_kwargs)
    elif act_type == "gelu":
        act_out = nn.GELU(**act_kwargs)
    elif act_type == "silu":
        act_out = nn.SiLU(**act_kwargs)
    elif act_type == "mish":
        act_out = nn.Mish(**act_kwargs)
    elif act_type == "tanh":
        act_out = nn.Tanh(**act_kwargs)

    return act_out


class MiniRefiner(nn.Module):
    def __init__(self, cfg, latent_channels: int = 4, aov_channels=0):
        super().__init__()

        self.conditioning_mode = cfg["conditioning_mode"]
        if self.conditioning_mode not in ["concat", "xattn"]:
            raise ValueError(
                f"Unknown conditioning mode `{self.conditioning_mode}`.  Must be one of [`concat`, `xattn`]."
            )

        self.pre_edge_conv = cfg["pre_edge_conv"] if "pre_edge_conv" in cfg else False

        self.latent_channels = latent_channels
        self.aov_channels = aov_channels
        self.hidden_channels = int(latent_channels * cfg["expand_ratio"])
        self.output_channels = latent_channels

        self.per_layer_residuals = cfg["per_layer_residuals"]
        self.depthwise_residuals = cfg["depthwise_residuals"]

        self.kernels = cfg["refiner_kernels"]
        self.num_layers = len(self.kernels)

        self.normalization = cfg["refiner_normalization"]
        identity_norm = self.normalization.lower() == "identity"
        self.activation = cfg["refiner_activation"]
        
        self.rcp_e = 1 / math.e

        self.dropout = nn.Dropout(p=0.1)

        # Downsample AOVs from full res to hidden channel res
        self.aov_downsampler = (
            nn.Sequential(
                nn.LazyConv2d(16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
            )
            if cfg["downsample_aovs"]
            else None
        )
        
        # Apply a single depthwise pre-convolution
        self.pre_depthwise_conv = nn.Sequential(
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1, groups=latent_channels, bias=identity_norm),
            get_norm_layer(self.normalization, latent_channels),
            get_activation_layer(self.activation),
        )

        # Hidden layer convolutions, the main body of the refiner
        self.hidden_convs = nn.ModuleList()
        for i, k in enumerate(self.kernels):
            in_channels = self.hidden_channels
            out_channels = self.hidden_channels
            # First layer: account for AOVs (16 channels if downsampled)
            if i == 0:
                in_channels = latent_channels + (16 if self.aov_downsampler is not None else aov_channels)

            self.hidden_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=k // 2, bias=identity_norm),
                    get_norm_layer(self.normalization, out_channels),
                    get_activation_layer(self.activation),
                )
            )

        # Output convolutions.  Either a single conv at the end, or per-layer outputs.
        self.out_convs = nn.ModuleList()
        for i in range(len(self.kernels)):
            is_last = i == len(self.kernels) - 1

            in_channels = self.hidden_channels
            out_channels = latent_channels

            # TODO better multi-residual handling for the new per-layer residual
            if is_last:
                out_channels = self.output_channels

            if self.per_layer_residuals or is_last:
                if self.depthwise_residuals:
                    c = nn.Sequential(
                        nn.Conv2d(self.hidden_channels, self.output_channels, kernel_size=1, bias=identity_norm),
                        get_norm_layer(self.normalization, out_channels),
                        get_activation_layer(self.activation),
                        nn.Conv2d(self.output_channels, self.output_channels, kernel_size=3, padding=k//2, bias=True, groups=self.output_channels)
                    )
                    nn.init.zeros_(c[-1].bias)
                else:
                    c = nn.Conv2d(
                        self.hidden_channels, self.output_channels, kernel_size=1, bias=True
                    )
                    nn.init.zeros_(c.bias)
                self.out_convs.append(c)
            else:
                self.out_convs.append(None)


    def forward(self, sample: torch.Tensor | list[torch.Tensor], aovs: torch.Tensor):
        # Prepare sample batch in case of multiple images
        if isinstance(sample, torch.Tensor):
            sample = [sample]
        sample = torch.cat(sample, dim=0)

        out = {}
        layer_residuals = []

        # Depthwise pre-convolution step
        if self.pre_edge_conv:
            pre_residual = self.pre_depthwise_conv(torch.tanh(sample * self.rcp_e))
            sample = sample + pre_residual
            out["pre_residual"] = pre_residual
            out["pre_sample"] = sample

        # Optional AOV conditioning.
        if aovs is not None:
            # If batching, duplicate AOVs to batches
            if aovs.shape[0] != sample.shape[0]:
                aovs = aovs.expand(sample.shape[0], -1, -1, -1)

            if self.aov_downsampler is not None:
                aovs = self.aov_downsampler(aovs)

            if self.conditioning_mode == "concat":
                x = torch.cat((sample, aovs), dim=1)
            elif self.conditioning_mode == "xattn":
                raise NotImplementedError(
                    "Cross-attention has been temporarily removed.  It provides more complexity and little benefit."
                )
            else:
                raise RuntimeError(
                    "AOV conditioning was enabled, but no valid conditioning mode was set!  Please set `conditioning_mode` to [`concat`, `xattn`]."
                )
        else:
            x = sample

        # Input scaling for stability
        x = torch.tanh(x * self.rcp_e)

        # Main refinement loop
        for i, conv in enumerate(self.hidden_convs):
            is_last = i == len(self.hidden_convs) - 1
            out_conv = self.out_convs[i]
    
            x = conv(x)

            # Should this layer have an output residual?
            if self.per_layer_residuals or is_last:
                res = out_conv(x)
                layer_residuals.append(res)

            x = self.dropout(x)

        out["base"] = layer_residuals[-1]
        if self.pre_edge_conv:
            out["base"] = out["base"] + out["pre_residual"]

        # Handle per-layer residuals
        if self.per_layer_residuals:
            out["layer_residuals"] = layer_residuals
            out["layer_samples"] = [sample + r for r in layer_residuals]

        return out
    
    def get_config(self):
        return {
            "latent_channels": self.latent_channels,
            "aov_channels": self.aov_channels,
            "expand_ratio": self.hidden_channels / self.latent_channels,
            "refiner_activation": self.activation,
            "refiner_normalization": self.normalization,
            "conditioning_mode": self.conditioning_mode,
            "pre_edge_conv": self.pre_edge_conv,
            "per_layer_residuals": self.per_layer_residuals,
            "depthwise_residuals": self.depthwise_residuals,
            "downsample_aovs": self.aov_downsampler is not None,
            "refiner_kernels": self.kernels,
        }
