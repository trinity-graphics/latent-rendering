from pathlib import Path

import torch
from torch import nn
from wandb.sdk.wandb_config import Config

from utils import img_utils

from .mini_refiner import MiniRefiner


class NetworkManager(nn.Module):
    def __init__(self, cfg: Config, device: torch.device, latent_channels, aov_channels = 0):
        super().__init__()
        self.use_refiner = cfg["use_refiner"]
        self.miniRefiner = None

        # Debugging
        self.debug_outputs = set()
        self.debug_out_dir = None

        if self.use_refiner:
            self.miniRefiner = MiniRefiner(
                cfg,
                latent_channels=latent_channels,
                aov_channels=aov_channels,
            ).to(device)

    def save_debug_image(
        self, image, name: str, debug: bool | int, save_exr: bool = False
    ):
        if debug is not False:
            if self.debug_out_dir is None:
                raise RuntimeError(
                    "save_debug_image(...) was called without setting an output directory.  Call set_debug_out_dir(...) from the RendererManager first."
                )

            # Track debug output names
            self.debug_outputs.add(name)

            if isinstance(debug, int) and not isinstance(debug, bool):
                suffix = f"_{debug}"
            else:
                suffix = ""

            fname = str(self.debug_out_dir / (name + suffix))

            save_fn = None
            if save_exr:
                save_fn = img_utils.save_latent_exr
                save_fn(image, fname)

            save_fn = img_utils.save_latent_png
            save_fn(image, fname)

    def get_debug_outputs(self):
        return self.debug_outputs

    def refine_latent_conv(self, latent, aovs=None, debug=False):
        residuals = self.miniRefiner(latent, aovs)
        latent = latent[0] + residuals["base"]
        return latent, residuals

    def set_debug_out_dir(self, path: Path):
        self.debug_out_dir = path

    def is_refiner_enabled(self) -> bool:
        return self.use_refiner

    def load_checkpoint(self, path: Path):
        if not path.exists():
            return
        data = torch.load(str(path), map_location="cpu", weights_only=False)
        refiner_state = data.get("refiner")

        if self.is_refiner_enabled() and refiner_state is not None:
            missing, unexpected = self.miniRefiner.load_state_dict(refiner_state, strict=False)
            if missing or unexpected:
                print("[MISSING/UNEXPECTED] Refiner: ", missing, unexpected)
