from enum import StrEnum

import drjit as dr
import mitsuba as mi

from utils import sensor_utils

from .base import Renderer


class AOVRenderer(Renderer):
    class AOVTypes(StrEnum):
        albedo = "al:albedo"
        depth = "dd:depth"
        position = "po:position"
        uv = "co:uv"
        geo_normal = "gn:geo_normal"
        sh_normal = "sn:sh_normal"
        prim_index = "pi:prim_index"
        shape_index = "si:shape_index"
        dp_du = "pu:dp_du"
        dp_dv = "pv:dp_dv"
        duv_dx = "ux:duv_dx"
        duv_dy = "uy:duv_dy"

    def __init__(self, cfg, ref_sensor, res, ch):
        super().__init__()
        self.mi_variant = f"{mi.variant().split('_')[0]}_ad_rgb"
        self.resolution = res
        self.channels = len(cfg["aovs_list"].split(","))
        # https://github.com/mitsuba-renderer/mitsuba3/issues/1189#issuecomment-2146822617
        # The link above discusses an issue with using AOVs alongside spectral rendering.
        # We circumvent this by loading a standalone AOV integrator and using a separate render call.
        self.spp = cfg["samples_per_pixel_aovs"]
        with mi.variant_context(self.mi_variant):
            self.aov_integrator = mi.load_dict(
                {"type": "aov", "aovs": cfg["aovs_list"]}
            )
            self.sensor = sensor_utils.load_sensor_like(ref_sensor)
            sensor_utils.set_rgb_film(self.sensor, res=res)
            self.sensor_params = mi.traverse(self.sensor)

    def set_aovs(self, aovs: list[AOVTypes]):
        with mi.variant_context(self.mi_variant):
            self.aov_integrator = mi.load_dict(
                {"type": "aov", "aovs": ','.join([str(a) for a in aovs])}
            )

    @dr.wrap(source="torch", target="drjit")
    def _render(self, scene, params, seed):
        with dr.suspend_grad(), mi.variant_context(self.mi_variant):
            return mi.render(
                scene,
                params=params,
                sensor=self.sensor,
                integrator=self.aov_integrator,
                spp=self.spp,
                seed=mi.UInt32(seed),
            )

    def render(self, scene, params, seed=0, debug=False):
        # Render AOVs
        aovs = self._render(scene, params, seed)

        # Prepare for use with Pytorch by transforming HWC->BCHW
        aovs = aovs.permute(2, 0, 1).unsqueeze(0)

        self.save_debug_image(aovs, "aovs", debug)
        return aovs