import numpy as np
import torch
from diffusers import AutoencoderKL, AutoencoderKLQwenImage, AutoencoderKLWan
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution


class MiVae:
    r"""
    Wrapper class for the Stable Diffusion VAE, handles conversions from Mitsuba's
    image format to a VAE-acceptable shape.

    Args:
        model_name (`str`, *optional*, defaults to `stabilityai/stable-diffusion-2-1`):
            The model from which the VAE will be used.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            The model datatype to be used.
        device (`torch.device`, *optional*, defaults to `torch.device("cpu")`):
            The device to move the VAE to.
    """

    def __init__(
        self,
        model_name: str = "stabilityai/stable-diffusion-2-1",
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ):
        self.model_name = model_name
        self.is_qwen_model = self.model_name.startswith("Qwen")
        self.is_wan_model = self.model_name.startswith("Wan")
        self.is_eq_model = self.model_name.startswith("zelaki") or self.model_name.startswith("KBlueLeaf")

        kw = {
            "torch_dtype": dtype
        }
        if self.is_qwen_model:
            AEClass = AutoencoderKLQwenImage
        elif self.is_wan_model:
            AEClass = AutoencoderKLWan
        else:
            AEClass = AutoencoderKL

        if not self.is_eq_model:
            kw["subfolder"] = "vae"

        self.vae = AEClass.from_pretrained(
            model_name, **kw
        ).to(device)
        
        self.vae.eval()

        if self.is_qwen_model:
            self.vae_scale_factor = (
                2 ** len(self.vae.temperal_downsample)
                if getattr(self, "vae", None)
                else 8
            )
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(device, dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                device, dtype
            )
            self.scaling_factor = latents_std
            self.shift_factor = latents_mean
        elif self.is_wan_model:
            self.vae_scale_factor = (
                self.vae.config.scale_factor_spatial 
                if getattr(self, "vae", None) 
                else 8
            )
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(self.vae.device, self.vae.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                self.vae.device, self.vae.dtype
            )
            self.scaling_factor = latents_std
            self.shift_factor = latents_mean
        else:
            self.vae_scale_factor = (
                2 ** (len(self.vae.config.block_out_channels) - 1)
                if getattr(self, "vae", None)
                else 8
            )
            self.scaling_factor = self.vae.config.scaling_factor
            self.shift_factor = self.vae.config.shift_factor

        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor * (2 if self.is_qwen_model else 1)
        )
        self.latent_dist = None

    def encode(self, image, return_parameters=False):
        r"""
        Encodes an RGB image into its compressed latent representation.

        Args:
            image (`mi.TensorXf`):
                The input image to encode.

        Returns:
            `mi.TensorXf`:
                The compressed latent sample, OR the distribution itself, depending on return_parameters.
        """
        # Generate latent distribution
        image = image.to(dtype=self.vae.dtype, device=self.vae.device)
        image = self.image_processor.preprocess(image).to(device=self.vae.device)
        # Needs additional "frame number" dimension for Qwen, as it uses a video VAE.
        if self.is_qwen_model or self.is_wan_model:
            image = image.unsqueeze(2)
        self._set_latent_distribution(self.vae.encode(image).latent_dist)

        # Either return the parameters of the distribution directly, or sample the distribution
        if return_parameters:
            out = torch.cat((self.latent_dist.mean, self.latent_dist.logvar), dim=1)
        else:
            out = self.latent_dist.sample()
            if self.shift_factor is not None:
                out = out - self.shift_factor
            out = out * self.scaling_factor

        if self.is_qwen_model or self.is_wan_model:
            out = out.squeeze(2)

        return out

    def decode(self, latents):
        r"""
        Decodes a latent to a full-resolution image.

        Args:
            latents (`mi.TensorXf`):
                The input latent to decode.

        Returns:
            `mi.TensorXf`:
                The decoded final image.
        """
        latents = latents.to(self.vae.device)
        if self.is_qwen_model or self.is_wan_model:
            latents = latents.unsqueeze(2)

        latents = latents / self.scaling_factor
        if self.shift_factor is not None:
            latents = latents + self.shift_factor

        latents = latents.to(dtype=self.vae.dtype, device=self.vae.device)
        image = self.vae.decode(latents).sample

        if self.is_qwen_model or self.is_wan_model:
            image = image.squeeze(2)
        image = self.image_processor.postprocess(image, output_type="pt")
        return image

    def preprocess_parameters(self, parameters):
        r"""
        Takes the parameters and returns the mean and the variance

        Args:
            image(`mi.TensorXf`):
                The input parameter image

        Returns:
            mean(mi.TensorXf), std(mi.TensorXf`):
                The mean and standard deviation from the distribution
        """
        mean, log_var = torch.chunk(parameters, chunks=2, dim=1)
        return mean, log_var

    def scale_mean(self, mean):
        if self.is_qwen_model:
            mean = mean.unsqueeze(2)
        if self.shift_factor is not None:
            out = mean - self.shift_factor
        else:
            out = mean
        out = out * self.scaling_factor
        if self.is_qwen_model:
            out = out.squeeze(2)
        return out


    def NLL_Loss(self, sample: torch.Tensor) -> torch.Tensor:
        if self.latent_dist.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi
            + self.latent_dist.logvar
            + torch.pow(sample - self.latent_dist.mean, 2) / self.latent_dist.var
        )


    def nll(self, sample):
        r"""
        A wrapper to return the NLL of a sample under the latent_dist
        generated by encode() or manually set with set_latent_distribution_parameters().

        Args:
            sample (`mi.TensorXf`):
                The sample to take the NLL of.

        Returns:
            `float`:
                The negative-log-likelihood of a sample under the saved latent distribution.
        """
        if self.latent_dist is None:
            raise ValueError(
                "No latent distribution saved. Call encode() or set_latent_distribution_parameters() first."
            )

        sample = sample / self.scaling_factor
        if self.shift_factor is not None:
            sample = sample + self.shift_factor
        return self.NLL_Loss(sample)

    def kl(self, mean, logvar):
        r"""
        A wrapper to return the KL Divergence of the distribution defined by mean and logvar against
        the latent_dist generated by encode() or manually set with set_distribution_parameters().

        Args:
            mean (`mi.TensorXf`):
                The mean of the distribution.
            logvar (`mi.TensorXf`):
                The log-variance of the distribution.

        Returns:
            `float`:
                The KL Divergence of a given distribution against the saved latent_dist.
        """
        if self.latent_dist is None:
            raise ValueError(
                "No latent distribution saved. Call encode() or set_latent_distribution_parameters() first."
            )

        other = self._init_latent_distribution(mean, logvar)
        return self.latent_dist.kl(other)

    def sample_latent_distribution(self, mean, logvar):
        r"""
        Samples a given mean and var.
        Does NOT update the internally stored latent_dist.

        Args:
            mean (`mi.TensorXf`):
                The mean of the distribution.
            logvar (`mi.TensorXf`):
                The log-variance of the distribution.
        """
        dist = self._init_latent_distribution(mean, logvar)
        sample = dist.sample()
        if self.shift_factor is not None:
            sample = sample - self.shift_factor
        sample = sample * self.scaling_factor
        if self.is_qwen_model:
            sample = sample.squeeze(2)
        return sample

    def set_latent_distribution(self, mean, logvar):
        r"""
        Manually update the saved latent distribution with a mean and log-variance.

        Args:
            mean (`mi.TensorXf`):
                The mean of the distribution.
            logvar (`mi.TensorXf`):
                The log-variance of the distribution.
        """
        dist = self._init_latent_distribution(mean, logvar)
        self._set_latent_distribution(dist)

    def _init_latent_distribution(self, mean, logvar):
        r"""
        HWC -> BCHW DiagonalGaussianDistribution __init__ wrapper.
        """
        params = torch.cat((mean, logvar), dim=1)
        if self.is_qwen_model:
            params.unsqueeze(2)
        return DiagonalGaussianDistribution(parameters=params, deterministic=False)

    def _set_latent_distribution(self, dist: DiagonalGaussianDistribution):
        self.latent_dist = dist

    def latent_channels(self):
        if self.is_qwen_model:
            return self.vae.config.z_dim
        return self.vae.config.latent_channels
    
    def latent_resolution(self, rgb_resolution):
        return rgb_resolution // self.vae_scale_factor
