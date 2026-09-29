from dataclasses import dataclass

import drjit as dr
import mitsuba as mi
import torch

from latents.latent_processor import LatentProcessor
from utils import io_utils, sensor_utils

from .base import Renderer


@dataclass
class RenderDebug:
    """Individually tone-mapped render terms, for debug/visualisation only.

    These are *not* combined or post-processed; the ready-to-use latent is
    returned separately by :meth:`LatentRenderer.render`.
    """
    latent: torch.Tensor   # main latent term, tone-mapped
    ambient: torch.Tensor  # ambient term, tone-mapped


class LatentRenderer(Renderer):
    def __init__(self, cfg, ref_sensor, res, ch):
        super().__init__()

        # Perform dummy postprocess operation to get the final shape
        self.resolution = res
        dummy = torch.zeros(1, ch, res, res)
        self.latent_processor = LatentProcessor(
            cfg,
            denoiser=mi.OptixDenoiser(input_size=(self.resolution, self.resolution)),
            latent_ch=ch,
        )
        dummy = self.latent_processor.preprocess_latent_to_render(dummy)
        self.channels = dummy.shape[1]

        self.scene = io_utils.load_file_as_latent(
            cfg["scene_file"],
            cfg=cfg,
            force_regenerate=True,
            # XML kwargs
            res=self.resolution,
            max_depth=cfg["max_depth"],
        )
        self.params = mi.traverse(self.scene)

        ### Flush out the shape params to the nodes
        # with open(os.devnull, 'w') as f:
        #     print(self.scene.emitters(), file=f)

        # self.params = mi.traverse(self.scene)

        self.sensor = sensor_utils.load_sensor_like(ref_sensor)
        sensor_utils.set_spectral_film(
            self.sensor,
            res=self.resolution,
            channels=self.channels,
            generate_spd=cfg["generate_spd"],
            generate_spd_gap=cfg["generate_spd_gap"],
        )
        self.sensor_params = mi.traverse(self.sensor)
        self.spp = cfg["samples_per_pixel_latents"]

        self.clip_rendered = cfg["clip_rendered"]
        self.clip_processed = cfg["clip_processed"]
        
        self.separate_term = cfg["seperate_term"]
        self.ambient_term = cfg["ambient_term"]

        # The latent integrator appends a trailing `missw` AOV whose product with
        # `miss_color` forms the flat background (applied in render). Absent for
        # the plain `prb` path, so key off whether the parameter exists.
        self.miss_color_key = next(
            (k for k, _ in self.params if "miss_color" in k), None
        )

    @dr.wrap(source="torch", target="drjit")
    def _render(self, seed, diff_params=None):
        if diff_params is not None:
            self.params.update(diff_params)
        return mi.render(
            self.scene,
            self.params,
            sensor=self.sensor,
            spp=self.spp,
            seed=seed,
            seed_grad=seed + 1,
        )
    
    def _miss_color(self, opt_params):
        """Per-channel flat background colour, shaped to broadcast as (1, C, 1, 1).

        Differentiable when ``miss_color`` is being optimised (present in
        ``opt_params``); otherwise read as a constant from the scene parameters.
        """
        key = self.miss_color_key
        if opt_params is not None and key in opt_params:
            mc = opt_params[key].as_subclass(torch.Tensor)
        else:
            mc = self.params[key].torch()
        return mc.reshape(1, -1, 1, 1)

    def _split_miss(self, rendered, opt_params):
        """Peel the trailing `missw` AOV off ``rendered`` and form the flat background.

        The integrator appends a detached `missw` AOV (throughput reaching gated
        misses); the flat background ``missw * miss_color`` is where `miss_color`
        becomes differentiable, applied in render() at image resolution. Returns
        ``(rendered_without_missw, miss_bg)`` with ``miss_bg`` as a (1, C, H, W)
        term ready to add in linear space, or ``None`` for the plain `prb` path.
        """
        if self.miss_color_key is None:
            return rendered, None
        mi_ch = len(mi.Spectrum())
        miss_weight = rendered[..., -mi_ch:].permute(2, 0, 1).unsqueeze(0)
        miss_bg = miss_weight * self._miss_color(opt_params)
        return rendered[..., :-mi_ch], miss_bg

    def render(self, seed=0, opt_params=None, debug=False):
        """Render the scene into a single, ready-to-use latent.

        The separate radiance terms are combined in linear space, tone-mapped,
        then post-processed for decoding (combine -> tonemap -> postprocess).

        Returns:
            (final_latent, debug): ``final_latent`` is the combined latent,
            ready to refine/decode. ``debug`` is a :class:`RenderDebug` holding
            the individually tone-mapped terms for debug/visualisation.
        """
        # dr.wrap needs torch.Tensors. `miss_color` is excluded: the integrator
        # doesn't read it (it only emits the `missw` AOV) -- the flat background
        # is applied below in torch, at image resolution.
        opt_tensors = None
        if opt_params is not None:
            opt_tensors = {
                k: p.as_subclass(torch.Tensor)
                for k, p in opt_params.items()
                if k != self.miss_color_key
            }
        rendered = self._render(seed, opt_tensors)
        
        # Split off the flat miss color
        rendered, miss_bg = self._split_miss(rendered, opt_params)

        # With `separate_term`, the integrator emits the ambient term as its own
        # AOV after the main latent; otherwise it is folded into the main chunk.
        terms = ["latent"]
        if self.separate_term and self.ambient_term:
            terms.append("ambient")
        chunks = torch.chunk(rendered, len(terms), dim=-1) if len(terms) > 1 else (rendered,)
        parts = {t: c.permute(2, 0, 1).unsqueeze(0) for t, c in zip(terms, chunks)}

        latent = parts["latent"]
        ambient = parts.get("ambient", torch.zeros_like(latent))

        # Flat background, added to the main term in linear space (where
        # `miss_color` carries its gradient); see _split_miss above.
        if miss_bg is not None:
            latent = latent + miss_bg

        # Clip the raw (linear) terms if configured.
        if self.clip_rendered:
            latent = torch.clip(latent, 0, 1)
            ambient = torch.clip(ambient, 0, 1)
        self.save_debug_image(latent, "latent", debug, save_exr=True)

        # Final latent: combine in linear space, tonemap, then postprocess for decode.
        final = latent + ambient
        final = self.latent_processor.apply_tonemap(final)
        final = self.latent_processor.postprocess_rendered_latent_to_decode(final)
        if self.clip_processed:
            final = torch.clip(final, -5, 5)
        self.save_debug_image(final, "processed", debug, save_exr=True)

        # Debug outputs, mapped back to the latent's channel layout (only differs with split/invert).
        lp = self.latent_processor
        debug_out = RenderDebug(
            latent=lp._join(lp._reverse_invert(lp.apply_tonemap(latent))),
            ambient=lp._join(lp._reverse_invert(lp.apply_tonemap(ambient))),
        )
        self.save_debug_image(debug_out.ambient, "ambient", debug, save_exr=True)

        return final, debug_out