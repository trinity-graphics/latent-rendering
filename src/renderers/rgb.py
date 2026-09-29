import drjit as dr
import mitsuba as mi
import torch

from utils import img_utils, sensor_utils

from .base import Renderer


class RGBRenderer(Renderer):
    def __init__(self, cfg, ref_sensor, res, ch):
        super().__init__()

        self.resolution = res
        self.channels = ch
        self.mi_variant = f"{mi.variant().split('_')[0]}_ad_rgb"
        # self.mi_variant = "scalar_rgb"

        with mi.variant_context(self.mi_variant):
            self.scene = mi.load_file(
                cfg["scene_file"],
                optimize=False,  # optimize=True can mess up custom plugins
                res=self.resolution,
                integrator="prb",
                max_depth=cfg["max_depth"],
            )
            self.params = mi.traverse(self.scene)
            self.sensor = sensor_utils.load_sensor_like(ref_sensor)
            sensor_utils.set_rgb_film(self.sensor, res=self.resolution)
            self.sensor_params = mi.traverse(self.sensor)
        self.spp = cfg["samples_per_pixel"]
        self.tone_map = cfg["ref_tone_map"]
        if cfg["denoise_references"]:
            self.denoiser = mi.OptixDenoiser(
                input_size=(self.resolution, self.resolution)
            )

    @dr.wrap(source="torch", target="drjit")
    def _render(self, seed, denoiser = True):
        with dr.suspend_grad(), mi.variant_context(self.mi_variant):
            img = mi.render(
                self.scene,
                self.params,
                sensor=self.sensor,
                spp=self.spp,
                seed=mi.UInt32(seed),
            )
            if denoiser:
                img = self.denoiser(img)
        return img

    def render(self, seed=0, debug=False, denoiser=True):
        # Render RGB image
        rgb = self._render(seed, denoiser=denoiser)

        # Prepare for use with Pytorch by transforming HWC->BCHW
        rgb = rgb.permute(2, 0, 1).unsqueeze(0)

        # Postprocess the RGB image
        if self.tone_map == "reinhard":
            rgb = img_utils.reinhard_tm(rgb)
        else:
            rgb = img_utils.clip_tm(rgb)

        self.save_debug_image(rgb, "rgb", debug)
        return rgb
    

class DiffRGBRenderer(Renderer):
    def __init__(self, cfg, ref_sensor, res, ch):
        super().__init__()

        if not mi.variant().endswith("ad_rgb"):
            raise RuntimeError("Requires use of an _ad_rgb Mitsuba variant.")

        self.resolution = res
        self.channels = ch

        self.scene = mi.load_file(
            cfg["scene_file"],
            optimize=False,  # optimize=True can mess up custom plugins
            res=self.resolution,
            integrator="prb",
            max_depth=cfg["max_depth"],
        )
        self.params = mi.traverse(self.scene)
        self.sensor = sensor_utils.load_sensor_like(ref_sensor)
        sensor_utils.set_rgb_film(self.sensor, res=self.resolution)
        self.sensor_params = mi.traverse(self.sensor)
        self.spp = cfg["samples_per_pixel"]
        self.tone_map = cfg["ref_tone_map"]

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
            seed_grad=seed+1,
        )
    
    def render(self, seed=0, opt_params=None, debug=False):
        opt_tensors = None
        if opt_params is not None:
            opt_tensors = {
                k: p.as_subclass(torch.Tensor) for k, p in opt_params.items()
            }
        
        # Render RGB image
        rgb = self._render(seed, opt_tensors)

        # Prepare for use with Pytorch by transforming HWC->BCHW
        rgb = rgb.permute(2, 0, 1).unsqueeze(0)

        eps = 1e-6
        rgb = torch.clip(rgb, eps, 1-eps)
        rgb = torch.pow(rgb, 1/2.2)

        self.save_debug_image(rgb, "rgb", debug)
        return rgb
