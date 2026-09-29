from dataclasses import dataclass, fields

import drjit as dr
import kornia
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
import torch
from scipy.stats import norm


@dataclass
class LatentData:
    lat_min: torch.Tensor | None = None
    lat_max: torch.Tensor | None = None
    offset: torch.Tensor | None = None
    mu: torch.Tensor | None = None
    sigma: torch.Tensor | None = None
    was_split: bool = None
    was_inverted: bool = False
    was_inverted_in_place: bool = False
    hist: torch.Tensor | None = None
    hist_bins: torch.Tensor | None = None
    reference: torch.Tensor | None = None

    def detach_all(self):
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, torch.Tensor):
                setattr(self, f.name, val.detach())


class LatentProcessor:
    def __init__(
        self,
        config,
        denoiser,
        latent_ch: int = 4,
    ):
        # Processor config
        self.offset_mode = config["offset_mode"]
        self.split_channels = config["split_channels"]
        self.additive_join_channels = config["additive_join_channels"]
        self.normalize = config["normalize"]
        self.scaling_mode = config["scaling_mode"]
        self.invert = config["invert"]
        # self.invert_in_place=config["invert_in_place"]
        self.pre_sigmoid = config["pre_sigmoid"]
        self.post_logit = config["post_logit"]
        self.tone_map = config["tone_map"]

        if self.scaling_mode == "channel":
            self.reduce_dim = (0, 2, 3)
        elif self.scaling_mode == "single":
            self.reduce_dim = None
        else:
            self.reduce_dim = None

        self.blur_ksize = config["loss1_blur_ksize"]
        self.denoiser = denoiser

        # Preprocessing outputs
        self.latent_data = None
        self.suppress_ld_updates = False

        # Constants
        self.LATENT_CH = latent_ch
        self.HALF_CH = self.LATENT_CH // 2

    #
    # Latent offset
    #
    def _offset(self, latent):
        if self.offset_mode == "mean":
            offset = torch.mean(latent, dim=self.reduce_dim)
        elif self.offset_mode == "median":
            offset = torch.quantile(latent, q=0.5, dim=self.reduce_dim)
        else:
            offset = None

        if offset is not None:
            offset = offset.view(1, -1, 1, 1)
            latent = latent - offset

        if not self.suppress_ld_updates:
            self.latent_data.offset = offset

        return latent

    def _reverse_offset(self, latent):
        if self.latent_data.offset is not None:
            latent = latent + self.latent_data.offset
        return latent

    #
    # Latent splitting
    #
    def _split(self, latent):
        if not self.split_channels:
            return latent

        pos = torch.clamp(latent, min=0)
        neg = torch.clamp(-latent, min=0)

        latent = torch.cat((pos, neg), dim=1)

        if not self.suppress_ld_updates:
            self.latent_data.was_split = True

        return latent

    def _join(self, latent):
        if not self.latent_data.was_split:
            return latent

        ch = latent.shape[1]
        half = ch // 2
        pos = latent[:, :half, :, :]
        neg = latent[:, half:, :, :]

        if self.additive_join_channels:
            return pos - neg

        return torch.where(pos > neg, pos, -neg)

    #
    # Latent inversion
    #
    def _invert(self, latent):
        if not self.invert:
            return latent

        ch = latent.shape[1]

        lmin = torch.amin(latent, dim=self.reduce_dim, keepdim=False)
        lmax = torch.amax(latent, dim=self.reduce_dim, keepdim=False)
        inv = lmin + lmax

        if self.reduce_dim is None:
            inv = inv.view(1, 1, 1, 1)
        else:
            inv = inv.view(1, ch, 1, 1)

        out = []
        for i in range(0, ch, self.LATENT_CH):
            l = i
            r = i + self.LATENT_CH
            block = latent[..., l:r]
            # Double channel count by concatenating inverted values rather than overwriting.
            out.append(torch.cat((block, inv[l:r] - block), dim=1))

        latent = torch.cat(out, dim=1)

        if not self.suppress_ld_updates:
            self.latent_data.was_inverted = True
        return latent

    def _reverse_invert(self, latent):
        if not self.latent_data.was_inverted:
            return latent

        # Should we be storing inv values from _invert and using them here rather than recalculating?
        ch = latent.shape[1]

        lmin = torch.amin(latent, dim=self.reduce_dim, keepdim=False)
        lmax = torch.amax(latent, dim=self.reduce_dim, keepdim=False)
        inv = lmin + lmax

        if self.reduce_dim is None:
            inv = inv.view(1, 1, 1, 1)
        else:
            inv = inv.view(1, ch, 1, 1)

        out = []
        for i in range(0, ch, 2 * self.LATENT_CH):
            l = i
            r = i + self.LATENT_CH
            il = r
            ir = r + self.LATENT_CH

            block = latent[..., l:r]
            inv_block = inv[il:ir] - latent[..., il:ir]
            out.append(block + inv_block)

        latent = torch.cat(out, dim=-1)
        return latent

    #
    # Latent normalization
    #
    def _normalize(self, latent):
        if not self.normalize:
            return latent

        ch = latent.shape[1]

        lat_min = latent.amin(dim=self.reduce_dim, keepdim=False)
        lat_max = latent.amax(dim=self.reduce_dim, keepdim=False)

        if self.reduce_dim is None:
            lat_min = lat_min.view(1, 1, 1, 1)
            lat_max = lat_max.view(1, 1, 1, 1)
        else:
            lat_min = lat_min.view(1, ch, 1, 1)
            lat_max = lat_max.view(1, ch, 1, 1)

        latent = (latent - lat_min) / (lat_max - lat_min)

        if not self.suppress_ld_updates:
            self.latent_data.lat_min = lat_min
            self.latent_data.lat_max = lat_max
        return latent

    def _denormalize(self, latent):
        if self.latent_data.lat_min is None or self.latent_data.lat_max is None:
            return latent

        latent = self.latent_data.lat_min + latent * (
            self.latent_data.lat_max - self.latent_data.lat_min
        )
        return latent

    def _sigmoid(self, latent):
        if not self.pre_sigmoid:
            return latent

        latent = torch.sigmoid(latent)
        return latent

    def _logit(self, latent):
        if not self.post_logit:
            return latent

        latent = torch.clip(latent, 0, 1)
        latent = torch.logit(latent, eps=1e-5)
        return latent

    def apply_tonemap(self, tensor):
        ## KEEP THEM DIFFERENT VALUES!
        ## Grad graph breaks if not
        ## The initial value being 0 messes this
        ## We need them to have different init values
        eps_pos = 1e-6
        eps_neg = 2e-6
        if self.tone_map == "reinhard":
            scale = 4
            pos_tensor = torch.clip(tensor, min=0)
            neg_tensor = torch.clip(-tensor, min=0)
            pos_val = pos_tensor / (1 + pos_tensor)
            neg_val = neg_tensor / (1 + neg_tensor)
            pos_scaled = torch.pow(pos_val + eps_pos, 1 / 2.2)
            neg_scaled = torch.pow(neg_val + eps_neg, 1 / 2.2)
            result = pos_scaled - neg_scaled
            return scale * result
        elif self.tone_map == "clip":
            scale = 3
            pos_tensor = torch.clip(tensor, min=0, max=scale)
            neg_tensor = torch.clip(-tensor, min=0, max=scale)
            pos_scaled = torch.pow(pos_tensor + eps_pos, 1 / 2.2)
            neg_scaled = torch.pow(neg_tensor + eps_neg, 1 / 2.2)
            result = scale * (pos_scaled - neg_scaled)
            return result
        else:
            return tensor
    
    def remove_tonemap(self, tensor):
        if self.tone_map == "reinhard":
            scale = 4
            pos_tensor = torch.clip(tensor, min=0)
            neg_tensor = torch.clip(-tensor, min=0)
            pos_scaled = torch.pow(pos_tensor / scale, 2.2)
            neg_scaled = torch.pow(neg_tensor / scale, 2.2)
            pos_val = pos_scaled / (1 - pos_scaled)
            neg_val = neg_scaled / (1 - neg_scaled)
            result = pos_val - neg_val
            return result
        if self.tone_map == "clip":
            scale = 3
            scaled_val = 3 ** (3.2 / 2.2)
            pos_tensor = torch.clip(tensor, min=0, max=scaled_val)
            neg_tensor = torch.clip(-tensor, min=0, max=scaled_val)
            pos_scaled = torch.pow(pos_tensor / scale, 2.2)
            neg_scaled = torch.pow(neg_tensor / scale, 2.2)
            result = pos_scaled - neg_scaled
            return result
        else:
            return tensor

    def plot_tensor_hist_with_gaussian(
        self, tensor, fname="out.png", bins=100, color="skyblue", alpha=0.6
    ):
        """
        Plots a histogram of a PyTorch tensor with a Gaussian fit overlay.

        Args:
            tensor (torch.Tensor): Input 1D tensor.
            bins (int): Number of histogram bins.
            color (str): Color of the histogram bars.
            alpha (float): Transparency of histogram bars.
        """
        data = tensor.detach().cpu().numpy()

        # Plot histogram
        plt.figure(figsize=(8, 5))
        plt.hist(
            data,
            bins=bins,
            density=True,
            alpha=alpha,
            color=color,
            edgecolor="black",
            label="Histogram",
        )

        # Fit and plot Gaussian
        mu, std = norm.fit(data)
        x = np.linspace(min(data), max(data), 100)
        p = norm.pdf(x, mu, std)
        plt.plot(
            x, p, "r", linewidth=2, label=f"Gaussian Fit\nμ = {mu:.2f}, σ = {std:.2f}"
        )

        plt.title("Tensor Histogram with Gaussian Fit")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(fname)

    def preprocess_latent_to_render(self, latent):
        r"""
        Prepares a latent with several postprocessing steps applied to make it easier to approximate with rendering.

        See the README for a full list of preprocessing configuration options.
        """
        self.latent_data = LatentData()

        latent = self._sigmoid(latent)
        latent = self._offset(latent)
        latent = self._split(latent)
        latent = self._normalize(latent)
        latent = self._invert(latent)

        self.latent_data.detach_all()
        return latent

    def postprocess_rendered_latent_to_decode(self, latent):
        r"""
        Prepares a rendered latent for decode by performing a series of configurable
        steps, including manual and neural postprocessing.

        Domain [0, 1]

        Returns a postprocessed latent.
        """
        if self.latent_data is None:
            raise ValueError(
                "LatentData not initialized, call preprocess_latent_to_render or read_latent_data first."
            )

        latent = self._reverse_invert(latent)
        latent = self._denormalize(latent)
        latent = self._join(latent)
        latent = self._reverse_offset(latent)
        latent = self._logit(latent)

        return latent

    def blur_latent(self, latent):
        latent = kornia.filters.median_blur(
            latent, kernel_size=(self.blur_ksize, self.blur_ksize)
        )
        return latent

    def denoise_latent(self, latent):
        B, C, H, W = latent.shape

        latent = mi.TensorXf(latent.squeeze(0).permute(1, 2, 0))
        # H, W, C = latent.shape
        latent_norm = (latent + 5) / 10

        channels = []
        for c in range(C):
            ch = latent_norm[..., c]
            ch = dr.reshape(ch, (H, W, 1))
            dup_ch = dr.repeat(ch, 3)
            dup_ch = dr.reshape(dup_ch, (H, W, 3))
            den_ch = self.denoiser(dup_ch)
            den = dr.reshape(den_ch[..., 0], (H, W, 1))
            channels.append(den)
        denoised_latent = dr.concat(channels, axis=-1)

        lat_denorm = (denoised_latent * 10) - 5

        lat_denorm = torch.as_tensor(latent).permute(2, 0, 1).unsqueeze(0)
        return lat_denorm
