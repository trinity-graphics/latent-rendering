from __future__ import annotations

from dataclasses import fields
from typing import TYPE_CHECKING

import torch

from networks import NetworkManager
from pipeline.mode import PipelineMode
from pipeline.outputs import (
    Latent,
    PipelineOutputs,
    PipelineTimingOutputs,
    Scalar,
    snapshot,
)
from renderers import RendererManager

if TYPE_CHECKING:
    from latents import MiVae


class LatentRenderingPipeline(torch.nn.Module):
    def __init__(self, cfg: dict, vae: MiVae, device=torch.device("cpu")):
        super().__init__()

        self._mode = PipelineMode.JOINT
        self.vae = vae

        rgb_res = cfg["image_resolution"]
        rgb_ch = 3
        latent_res = self.vae.latent_resolution(rgb_res)
        latent_ch = self.vae.latent_channels()

        self.rm = RendererManager(
            cfg,
            rgb_res,
            rgb_ch,
            latent_res,
            latent_ch,
        )

        #   Prepare Refiners
        aov_ch = (
            self.rm.aov.render(self.rm.rgb.scene, self.rm.rgb.params).shape[1]
            if cfg["use_aovs"]
            else 0
        )
        self.nets = NetworkManager(cfg, device, latent_ch, aov_ch)



    @property
    def num_refiner_layers(self) -> int:
        if not self.nets.is_refiner_enabled():
            return 0
        return self.nets.miniRefiner.num_layers
    
    @property
    def inactive_outputs(self) -> set[str]:
        capabilities = {
            "use_refiner": self.nets.use_refiner,
            "pre_edge_conv": bool(self.nets.miniRefiner and self.nets.miniRefiner.pre_edge_conv),
            "per_layer_residuals": bool(self.nets.miniRefiner and self.nets.miniRefiner.per_layer_residuals),
        }
        inactive = set()
        for cls in (PipelineOutputs, PipelineTimingOutputs):
            for f in fields(cls):
                if (meta := f.metadata.get("meta")) and meta.requires and not capabilities.get(meta.requires, True):
                    inactive.add(f.name)
        return inactive


    @property
    def mode(self) -> PipelineMode:
        return self._mode

    @mode.setter
    def mode(self, value: PipelineMode) -> None:
        self._mode = value

    def forward(
        self,
        seed=0,
        refs=None,
        cached_render=None,
        train_fns={},
        snap=False,
    ):
        
        out = PipelineOutputs()
        _s = snapshot if snap else lambda *a, **kw: None

        gt = refs.latent_sample if refs is not None else None
        decode_gt = refs.img if refs is not None else None

        # If in evaluation mode,
        # or training the scene,
        # or no cached render is provided,
        # render the latent.
        if not self.training or \
            PipelineMode.SCENE in self.mode or \
            cached_render is None:

            # TODO determine if scaling is really necessary
            params_for_render = self.rm.get_scaled_opt_params()

            latent, dbg = self.rm.latent.render(
                seed=seed, opt_params=params_for_render
            )
            out.processed = _s(Latent, dbg.latent, gt, decode_gt)
            out.ambient = _s(Latent, dbg.ambient)
            out.post_ambient = _s(Latent, latent, gt, decode_gt)

            if self.training:
                out.loss_1 = Scalar(data=train_fns["get_render_loss"](self.rm, refs, latent))
        else:
            latent, dbg = cached_render

        # If it evaluation mode,
        # or training the refiner,
        # pass the latent through the neural refiner.
        if not self.training or \
            PipelineMode.REFINER in self.mode:
            
            loss_2 = torch.tensor(0.0, device=self.vae.vae.device)
            residuals = None

            # Convolutional refiner
            if self.nets.is_refiner_enabled():
                if self.rm.aov is not None:
                    aovs = self.rm.aov.render(
                        self.rm.rgb.scene, self.rm.latent.params, seed=seed
                    )
                else:
                    aovs = None
                latent, residuals = self.nets.refine_latent_conv(latent, aovs)
                out.refined = _s(Latent, latent, gt, decode_gt)

                if self.nets.miniRefiner.per_layer_residuals:
                    out.layer_samples = []
                    out.layer_residuals = []
                    for i, l_s in enumerate(residuals["layer_samples"]):
                        out.layer_samples.append(_s(Latent, l_s, gt, decode_gt))
                    for i, l_r in enumerate(residuals["layer_residuals"]):
                        out.layer_residuals.append(_s(Latent, l_r))
                if self.nets.miniRefiner.pre_edge_conv:
                    out.pre_residual = _s(Latent, residuals["pre_residual"])
                    out.pre_sample = _s(Latent, residuals["pre_sample"], gt, decode_gt)

            if self.training:
                loss_2 += train_fns["get_refiner_loss"](refs, latent)
                loss_2 += train_fns["get_residual_losses"](
                    self.rm.latent.latent_processor, refs, residuals
                )
                out.loss_2 = Scalar(data=loss_2)

        out.final = _s(Latent, latent, gt, decode_gt)

        return out


    def time_forward(self, seed=0):
        # OLD TODO REMOVE 
        # timing = self.mode in PipelineMode.move_camera

        import torch.utils.benchmark as bench

        from testing.dr_timer import DrTimer

        out = PipelineTimingOutputs()

        params_for_render = self.rm.get_scaled_opt_params()
        out.time_latent_render = Scalar(data=torch.tensor(
            DrTimer(
                lambda: self.rm.latent.render(seed=seed, opt_params=params_for_render),
                warmup=1,
            ).timeit(1)
        ))

        latent, dbg = self.rm.latent.render(
            seed=seed, opt_params=params_for_render
        )

        if self.nets.is_refiner_enabled():
            if self.rm.aov is not None:
                aovs = self.rm.aov.render(
                    self.rm.rgb.scene, self.rm.latent.params, seed=seed
                )
            else:
                aovs = None

            def _refine_for_timing(lat):
                lat, _ = self.nets.refine_latent_conv(lat, aovs)
                return lat
            out.time_latent_refine = Scalar(data=torch.tensor(
                bench.Timer(
                    stmt="_refine(latent)",
                    globals={"_refine": _refine_for_timing, "latent": latent},
                ).timeit(1).mean * 1e3
            ))


        out.time_latent_decode = Scalar(data=torch.tensor(
            bench.Timer(
                stmt="vae.decode(lat)",
                globals={"vae": self.vae, "lat": latent},
            ).timeit(1).mean * 1e3
        ))

        return out
        

        