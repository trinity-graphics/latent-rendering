import numpy as np
import torch
import torch.nn.functional as F
from diffusers import (
    DDIMScheduler,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    UNet2DConditionModel,
)
from diffusers.training_utils import (
    compute_loss_weighting_for_sd3,
)


class SDSLoss:
    def __init__(
        self,
        cfg,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.model_name: str = cfg["model_id"]
        self.dtype: torch.dtype = dtype
        self.device: torch.device = device
        self.sds_lr = cfg["sds_lr"]
        self.sds_start = cfg["sds_start"]
        self.sds_stop = cfg["sds_stop"]
        self.sds_min_ts = cfg["sds_min_ts"]
        self.sds_max_ts = cfg["sds_max_ts"]
        self.sds_guidance_weight = cfg["sds_guidance_weight"]
        # self.embeddings = embeddings

        self.scheduler: DDIMScheduler = DDIMScheduler.from_pretrained(
            self.model_name, subfolder="scheduler", torch_dtype=self.dtype
        )
        self.unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
            self.model_name, subfolder="unet", torch_dtype=self.dtype
        ).to(device)

    def __call__(self, latent_in, embeddings, train_progress: float):
        with torch.no_grad():
            # Select a timestep
            sched_steps = self.scheduler.config.num_train_timesteps
            ts_int = np.clip(
                int((1 - train_progress) * sched_steps),
                self.sds_min_ts * sched_steps,
                self.sds_max_ts * sched_steps,
            )
            timestep = torch.tensor([ts_int], dtype=torch.int, device=self.device)

            # Inject noise according to timestep
            noise = torch.randn(*latent_in.shape, dtype=self.dtype, device=self.device)
            noisy_latent = self.scheduler.add_noise(latent_in, noise, timestep)

            # Get predicted noise from diffusion model
            null_embeddings = torch.zeros(
                *embeddings.shape, dtype=self.dtype, device=self.device
            )
            np_conditional = self.unet(
                noisy_latent, timestep=timestep, encoder_hidden_states=embeddings
            ).sample
            np_unconditional = self.unet(
                noisy_latent, timestep=timestep, encoder_hidden_states=null_embeddings
            ).sample
            prediction = np_unconditional + (self.sds_guidance_weight * (np_conditional - np_unconditional))

            # Calculate SDS loss
            alphas = self.scheduler.alphas_cumprod.type(self.dtype).to(self.device)
            weight = (1 - alphas[timestep]).view(-1, 1, 1, 1)
            noise_residual = weight * (prediction - noise)
            noise_residual = torch.nan_to_num(noise_residual)
        
        target = (latent_in - noise_residual).detach()
        loss = 0.5 * F.mse_loss(latent_in, target, reduction="sum")
        return loss

class SD3SDSLoss:
    def __init__(
        self,
        cfg,
        dtype: torch.dtype = torch.float16,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()
        self.model_id = cfg["model_id"]
        self.sds_min_ts = cfg["sds_min_ts"]
        self.sds_max_ts = cfg["sds_max_ts"]
        self.dtype = dtype
        self.device = device
        
        self.sds_guidance_weight = cfg.get("sds_guidance_weight", 7.5)
        
        # Load components
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.model_id, subfolder="scheduler"
        )
        self.transformer = SD3Transformer2DModel.from_pretrained(
            self.model_id, subfolder="transformer", torch_dtype=self.dtype
        ).to(device)
        self.transformer.eval()

    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        sigmas = self.scheduler.sigmas.to(device=self.device, dtype=dtype)
        schedule_timesteps = self.scheduler.timesteps.to(self.device)
        timesteps = timesteps.to(self.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    

    def __call__(self, latent_in, prompt_utils, train_progress: float):
        """
        latent_in: [H, W, 16] - The optimized parameters
        prompt_utils: Dictionary containing context and pooled_context (cond/uncond)
        """
        latent = latent_in
        with torch.no_grad():
            # 1. Setup Latents (Ensure 16 channels for SD3.5)
            # Permute from DrJit [H, W, C] to Torch [1, C, H, W]
            # latent = latent_in.clone().detach().to(dtype=self.dtype, device=self.device)
            
            # 2. Sample Timestep using Logit-Normal distribution (Standard for SD3 fine-tuning)
            # This ensures we sample the "middle" of the flow more frequently
            # u = torch.normal(mean=0.0, std=1.0, size=(1,), device=self.device).sigmoid()

            # u = compute_density_for_timestep_sampling(
            #     weighting_scheme="logit_normal",
            #     batch_size=1,
            #     logit_mean=0.0,
            #     logit_std=1.0,
            #     mode_scale=1.29,
            # )
            # indices = (u * self.scheduler.config.num_train_timesteps).long()
            # timestep = self.scheduler.timesteps[indices].to(self.device)
            # sigmas = self.get_sigmas(timestep, n_dim=latent.ndim, dtype=self.dtype)


            # Map u to discrete indices for the transformer's embedding layer
            sched_steps = self.scheduler.config.num_train_timesteps
            ts_idx = torch.as_tensor(np.clip(
                int(train_progress * sched_steps),
                self.sds_min_ts * sched_steps,
                self.sds_max_ts * sched_steps,
            ), dtype=torch.long, device=self.device)
            timestep = self.scheduler.timesteps[ts_idx].unsqueeze(0).to(device=self.device)
            sigmas = self.get_sigmas(timestep, n_dim=latent.ndim, dtype=self.dtype)

            # indices = (u * self.scheduler.config.num_train_timesteps).long()
            # timesteps = self.scheduler.timesteps[indices].to(self.device)

            # 3. Add Noise according to Flow Matching
            # zt = (1 - sigma) * x + sigma * noise
            noise = torch.randn_like(latent)
            noisy_latent = (1.0 - sigmas) * latent + sigmas * noise

            # 4. Predict Velocity with CFG
            # latent_model_input = torch.cat([noisy_latent] * 2)

            uncond_context = torch.zeros(*prompt_utils["context"].shape, dtype=self.dtype, device=self.device)
            uncond_pooled = torch.zeros(*prompt_utils["pooled_context"].shape, dtype=self.dtype, device=self.device)

            latent_model_input = torch.cat([noisy_latent] * 2)
            
            model_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=torch.cat([uncond_context, prompt_utils["context"]]),
                pooled_projections=torch.cat([uncond_pooled, prompt_utils["pooled_context"]]),
                return_dict=False,
            )[0]

            # Split for Classifier-Free Guidance
            v_uncond, v_cond = model_pred.chunk(2)
            v_pred = v_uncond + self.sds_guidance_weight * (v_cond - v_uncond)

            # preconditioning
            v_pred = v_pred * (-sigmas) + noisy_latent
            
            # SDS update: Weight * (Predicted Velocity - Target Velocity)
            # Standard weighting for RF-SDS often uses (1 - sigma)
            weighting = compute_loss_weighting_for_sd3("logit_normal", sigmas=sigmas)
            
        # 5. SDS Gradient Calculation
        # The target in SD3 flow matching is: target = noise - model_input
        # target = noise - latent
        target = latent
        noise_residual = weighting * (v_pred - target)
        noise_residual = torch.nan_to_num(noise_residual)

        # 6. Apply SDS update as a pseudo-loss
        # Squeeze and permute back to DrJit format [H, W, C]
        target_latent = (latent - noise_residual)
        loss = 0.5 * F.mse_loss(latent, target_latent, reduction="sum")

        return loss


# class SD3SDSLoss:
#     def __init__(
#         self,
#         cfg,
#         dtype: torch.dtype = torch.float16,
#         device: torch.device = torch.device("cuda"),
#     ):
#         super().__init__()
#         self.model_id = cfg["model_id"]
#         self.dtype = dtype
#         self.device = device
        
#         self.sds_guidance_weight = cfg.get("sds_guidance_weight", 7.5)
        
#         # Load components
#         self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
#             self.model_id, subfolder="scheduler"
#         )
#         self.transformer = SD3Transformer2DModel.from_pretrained(
#             self.model_id, subfolder="transformer", torch_dtype=self.dtype
#         ).to(device)
#         self.transformer.eval()

#     def get_sigmas(self, timesteps, n_dim, dtype):
#         """Standard SD3 sigma calculation from timesteps."""
#         sigmas = timesteps / self.scheduler.config.num_train_timesteps
#         while len(sigmas.shape) < n_dim:
#             sigmas = sigmas.unsqueeze(-1)
#         return sigmas.to(dtype)

#     def __call__(self, latent, prompt_utils, train_progress: float):
#         """
#         latent_in: [H, W, 16] - The optimized parameters
#         prompt_utils: Dictionary containing context and pooled_context (cond/uncond)
#         """
#         with torch.no_grad():
#             # 1. Setup Latents (Ensure 16 channels for SD3.5)
#             # Permute from DrJit [H, W, C] to Torch [1, C, H, W]
#             # latent = latent_in.clone().detach().to(dtype=self.dtype, device=self.device)
            
#             # 2. Sample Timestep using Logit-Normal distribution (Standard for SD3 fine-tuning)
#             # This ensures we sample the "middle" of the flow more frequently
#             u = torch.normal(mean=0.0, std=1.0, size=(1,), device=self.device).sigmoid()
            
#             # Map u to discrete indices for the transformer's embedding layer
#             num_train_steps = self.scheduler.config.num_train_timesteps
#             indices = (u * num_train_steps).long().cpu()
#             timesteps = self.scheduler.timesteps[indices].to(self.device)

#             # 3. Add Noise according to Flow Matching
#             # zt = (1 - sigma) * x + sigma * noise
#             noise = torch.randn_like(latent)
#             sigmas = self.get_sigmas(timesteps, n_dim=latent.ndim, dtype=self.dtype)
#             noisy_latent = (1.0 - sigmas) * latent + sigmas * noise

#             # 4. Predict Velocity with CFG
#             latent_model_input = torch.cat([noisy_latent] * 2)
            
#             model_pred = self.transformer(
#                 hidden_states=latent_model_input,
#                 timestep=timesteps.expand(2),
#                 encoder_hidden_states=torch.cat([prompt_utils["uncond_context"], prompt_utils["context"]]),
#                 pooled_projections=torch.cat([prompt_utils["uncond_pooled_context"], prompt_utils["pooled_context"]]),
#                 return_dict=False,
#             )[0]

#             # Split for Classifier-Free Guidance
#             v_uncond, v_cond = model_pred.chunk(2)
#             v_pred = v_uncond + self.sds_guidance_weight * (v_cond - v_uncond)
            
#             # SDS update: Weight * (Predicted Velocity - Target Velocity)
#             # Standard weighting for RF-SDS often uses (1 - sigma)
#             weighting = (1.0 - sigmas) 
            
#         # 5. SDS Gradient Calculation
#         # The target in SD3 flow matching is: target = noise - model_input
#         target = noise - latent
#         noise_residual = weighting * (v_pred - target)
#         # grad = torch.nan_to_num(grad)

#         # 6. Apply SDS update as a pseudo-loss
#         # Squeeze and permute back to DrJit format [H, W, C]
#         target_latent = (latent - noise_residual)
#         loss = 0.5 * F.mse_loss(latent, target_latent, reduction="sum")
#         return loss
