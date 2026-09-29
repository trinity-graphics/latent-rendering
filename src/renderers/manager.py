import logging
from collections.abc import Iterable
from dataclasses import fields
from itertools import product
from pathlib import Path

import drjit as dr
import mitsuba as mi
import torch
from torch.nn import Module, Parameter

from utils import img_utils

if mi.variant() is None:
    logging.warning("Renderers imported without setting a Mitsuba variant.  Renderers relying on custom plugins will not work.")
else:
    import utils.plugin_utils
    from utils import sensor_utils

from .aov import AOVRenderer
from .latent import LatentRenderer
from .rgb import DiffRGBRenderer, RGBRenderer


def linspace(min_val, max_val, num):
    if num == 1:
        return [(max_val + min_val) / 2]
    step = (max_val - min_val) / (num - 1)
    return [min_val + i * step for i in range(num)]





class RendererManager(Module):
    AMBIENT_TERMS = [
        "secondary_reflectance",
        "neg_secondary_reflectance",
        "occlusion_reflectance",
        "neg_occlusion_reflectance",
    ]

    REFL_TERMS = [
        ".tex_scale.value",
        ".tex_offset.value",
        ".diffuse_reflectance.value",
        ".reflectance.value",
        ".specular_reflectance.value",
        "specular_transmittance.value",
    ]

    def __init__(self, cfg, rgb_res, rgb_ch, latent_res, latent_ch):
        super().__init__()
        self.config = cfg

        # Store
        self.rgb_res = rgb_res
        self.rgb_ch = rgb_ch
        self.latent_res = latent_res
        self.latent_ch = latent_ch
        self.aov_res = rgb_res if cfg["downsample_aovs"] is True else latent_res
        self.aov_ch = -1  # Not known at this time, to be set dynamically later

        utils.plugin_utils.register_plugins(cfg)

        # Set up main sensor (for instantiating others)
        with mi.variant_context("cuda_ad_rgb"):
            self.main_scene_rgb = mi.load_file(
                cfg["scene_file"],
                optimize=False,
                res=cfg["image_resolution"],
                integrator="prb",
                max_depth=cfg["max_depth"],
            )
            self.main_sensor_rgb = self.main_scene_rgb.sensors()[0]
        self.main_scene_lat = mi.load_file(
            cfg["scene_file"],
            optimize=False,
            res=cfg["image_resolution"],
            integrator="prb",
            max_depth=cfg["max_depth"],
        )
        self.main_sensor_lat = self.main_scene_lat.sensors()[0]
        self.main_sensor_to_world = mi.ScalarTransform4f(
            self.main_sensor_rgb.world_transform().matrix.numpy().squeeze(-1)
        )

        # For generating references
        self.denoise_ref_lats = cfg["loss1_denoised_latent"]
        self.blur_ref_lats = cfg["loss1_blurred"]
        self.sample_ref_lats = cfg["train_on_dist_samples"]

        # Set up reference views
        self.ref_view_mode = cfg["move_mode"]
        self.light_arg = cfg["light_arg"]
        self.object_arg = cfg["object_arg"]
        self.light_object_arg = cfg.get("light_object_arg", None)
        self.light_object_0_arg = cfg.get("light_object_0_arg", None)
        get_ref_view_pos = {
            "camera": self.get_camera_view_positions,
            "light": self.get_light_view_positions,
            "object": self.get_object_view_positions,
        }[self.ref_view_mode]
        self.move_function = {
            "camera": self.rotate_sensors,
            "light": self.move_light,
            "object": self.move_object,
        }[self.ref_view_mode]
        self.view_positions = get_ref_view_pos(cfg)
        self.num_views = len(self.view_positions)
        self.main_view_idx = 0
        self.TRANSFORM_KEY = "to_world"

        self.negative_latent_rendering = cfg["negative_rendering"]

        # Prepare the actual renderers
        self.reset_renderers()
        self.scene_opt_params = {}

        # Handle primitive vs loaded model movement
        for target in ("light", "object", "light_object", "light_object_0"):
        # for target in ("light", "object"):
            tkey = getattr(self, f"{target}_arg")
            if tkey is None:
                setattr(self, f"{target}_key", None)
                setattr(self, f"{target}_data", None)
                setattr(self, f"{target}_move_fn", self.nullfn)
                continue

            if f"{tkey}.to_world" in self.rgb.params:
                mode = "prim"
            elif f"{tkey}.vertex_positions" in self.rgb.params:
                mode = "model"
            else:
                raise KeyError(f"No valid keys found for {target} target {tkey}")

            if mode == "prim":
                key = f"{tkey}.to_world"
                data = mi.Transform4f(self.rgb.params[key].matrix.numpy().squeeze(-1))
                move_fn = self.move_matrix
            elif mode == "model":
                key = f"{tkey}.vertex_positions"
                data = dr.unravel(mi.Point3f, self.rgb.params[key])
                move_fn = self.move_vertices

            setattr(self, f"{target}_key", key)
            setattr(self, f"{target}_data", data)
            setattr(self, f"{target}_move_fn", move_fn)


    def set_optimizable_parameters(
        self,
        param_substrings: Iterable[str]
    ):
        # Get scene params to optimize
        keys = []
        debug_str = "\n"
        for target in param_substrings:
            debug_str += f"Target Key: {target}\n"
            for p in self.latent.params:
                if target in p[0]:
                    keys.append(p[0])
                    debug_str += f"\tOptimizable Key: {p[0]}\n"

        num_parameters = len(keys)
        num_floats = num_parameters * self.latent.channels
        debug_str += f"{num_parameters} parameters tracked for a total of {num_floats} optimizable values."
        logging.info(debug_str)

        self.scene_opt_params = {
            k: Parameter(self.latent.params[k].torch().detach().clone())
            for k in keys
        }

    def get_scaled_opt_params(self) -> dict:
        result = {}
        for k, p in self.scene_opt_params.items():
            if any(ref_term in k for ref_term in self.REFL_TERMS):
                result[k] = torch.sigmoid(p)
            else:
                result[k] = p
        return result

    def set_debug_out_dir(self, path: Path):
        self.rgb.debug_out_dir = path
        self.latent.debug_out_dir = path
        if self.aov is not None:
            self.aov.debug_out_dir = path

    def get_debug_outputs(self):
        aov_outs = self.aov.debug_outputs if self.aov else set()
        return self.rgb.debug_outputs | self.latent.debug_outputs | aov_outs

    def reset_renderers(self):
        cfg = self.config
        if mi.variant().endswith("ad_rgb"):
            self.rgb = DiffRGBRenderer(cfg, self.main_sensor_rgb, self.rgb_res, self.rgb_ch)
        else:
            self.rgb = RGBRenderer(cfg, self.main_sensor_rgb, self.rgb_res, self.rgb_ch)
        try:
            self.latent = LatentRenderer(
                cfg, self.main_sensor_lat, self.latent_res, self.latent_ch
            )
        except NotImplementedError as e:
            print(f"[WARNING] Mitsuba is running in a variant that does not support latent rendering. RendererManager.latent will be set to None.\n{e}")
            self.latent = None
        self.aov = None
        if cfg["use_aovs"] is True:
            self.aov = AOVRenderer(cfg, self.main_sensor_rgb, self.aov_res, self.aov_ch)

    def get_camera_view_positions(self, cfg):
        x_range, y_range = cfg["x_range"], cfg["y_range"]
        x_views, y_views = cfg["x_views"], cfg["y_views"]
        # If using random/dataset renders, we will only have a single sensor, (0,0), for debug rendering.
        views = (
            []
            if (cfg["random_pos"] or cfg["dataset_training"])
            else list(
                product(
                    linspace(-(x_range / 2), x_range / 2, x_views),
                    linspace(-(y_range / 2), y_range / 2, y_views),
                )
            )
        )
        # Move "main view" to the front of the view list
        if (0, 0) in views:
            views.remove((0, 0))
        views.insert(0, (0, 0))
        return views

    def get_light_view_positions(self, cfg):
        x_views, y_views = cfg["x_views"], cfg["y_views"]
        # If using random/dataset renders, we will only have a single sensor, (0,0), for debug rendering.
        views = (
            []
            if (cfg["random_pos"] or cfg["dataset_training"])
            else list(
                product(
                    linspace(0, 1, x_views),
                    linspace(0, 1, y_views),
                )
            )
        )

        # Move "main view" to the front of the view list
        if (0.5, 0.5) in views:
            views.remove((0.5, 0.5))
        views.insert(0, (0.5, 0.5))

        offset = list()
        for view in views:
            offset.append(mi.Point3f((view[0] - 0.5) * 1.4, (view[1] - 0.5) * 1.4, 0))

        return offset

    def get_object_view_positions(self, cfg):
        x_range, y_range = cfg["x_range"], cfg["y_range"]
        x_views, y_views = cfg["x_views"], cfg["y_views"]

        views = list(range(0, y_range - y_range % y_views, y_range // y_views))

        offset = list()
        for view in views:
            offset.append(mi.Point3f(0, (view * 5) / y_range, 0))

        return offset

    def get_reference_views(self, get_refs_fn, out_dir):
        self.ref_views = []
        for n, v in enumerate(self.view_positions):
            self.move_function(*v)
            r = get_refs_fn(self)
            self.ref_views.append(r)

            # Save all reference views
            for n, view in enumerate(self.ref_views):
                for f in fields(view):
                    out = getattr(view, f.name)
                    if out is None:
                        continue
                    path = str(out_dir / "views" / f"{f.name}_{n}")
                    if f.name.startswith("latent"):
                        img_utils.save_latent_exr(out, path)
                        img_utils.save_latent_png(out, path)
                    else:
                        img_utils.save_image_exr(out, path)
                        img_utils.save_image_png(out, path)

    def get_rendered_views(self, out_dir, refiner_renders_per_view):
        self.rendered_views = []
        self.rendered_views_debug = []
        # Reduce SPP if using bootstrap aggregation
        view_seed = 0
        # self.latent.set_spp(self.latent.spp // refiner_renders_per_view)
        for n, v in enumerate(self.view_positions):
            self.move_function(*v)

            renders = []
            debugs = []
            with torch.no_grad(), dr.suspend_grad():
                for i in range(refiner_renders_per_view):
                    r, dbg = self.latent.render(seed=view_seed, opt_params=self.get_scaled_opt_params())
                    renders.append(r)
                    debugs.append(dbg)
                    view_seed += 1
            self.rendered_views.append(renders)
            self.rendered_views_debug.append(debugs)
            fname = "optim_rendered"
            path = str(out_dir / "views" / f"{fname}_{n}")
            img_utils.save_latent_exr(r, path)

    def rotate_sensors(self, az, el):
        for r in [self.rgb, self.latent, self.aov]:
            if r is None or r.sensor_params is None:
                continue
            r.sensor_params[self.TRANSFORM_KEY] = sensor_utils.get_rotated_transform(
                self.main_sensor_to_world, az, el
            )
            r.sensor_params.update()

    def rotate_move_sensor(self, az, el, x, y, z):
        for r in [self.rgb, self.latent, self.aov]:
            if r is None or r.sensor_params is None:
                continue
            r.sensor_params[self.TRANSFORM_KEY] = (
                sensor_utils.get_rotated_translated_transform(
                    self.main_sensor_to_world, az, el, mi.ScalarVector3f(x, y, z)
                )
            )
            r.sensor_params.update()

    def move_vertices(self, params, key, data, offset):
        V = mi.Point3f(data.x + offset.x, data.y + offset.y, data.z + offset.z)
        params[key] = dr.ravel(V)
        params.update()

    def move_matrix(self, params, key, data, offset):
        params[key] = data.translate(offset)
        params.update()

    def move_light(self, x, y, z):
        if self.light_arg is None:
            raise ValueError("No light was set to be tracked in the config!")
        offset = mi.Point3f(x, y, z)
        for r in [self.rgb, self.latent, self.aov]:
            if r is None or r.params is None:
                continue
            self.light_move_fn(r.params, self.light_key, self.light_data, offset)

    def move_object(self, x, y, z):
        if self.object_arg is None:
            raise ValueError("No object was set to be tracked in the config!")
        # if self.object_1_arg is None:
        #     raise ValueError("No object was set to be tracked in the config!")
        offset = mi.Point3f(x, y, z)
        for r in [self.rgb, self.latent, self.aov]:
            if r is None or r.params is None:
                continue
            self.object_move_fn(r.params, self.object_key, self.object_data, offset)
            # self.object_1_move_fn(r.params, self.object_1_key, self.object_1_data, offset)
    
    def nullfn(self, *args):
        pass
    
    def move_light_object(self, x, y, z):
        offset = mi.Point3f(x, y, z)
        for r in [self.rgb, self.latent, self.aov]:
            if r is None or r.params is None:
                continue
            self.light_object_move_fn(r.params, self.light_object_key, self.light_object_data, offset)
            self.light_object_0_move_fn(r.params, self.light_object_0_key, self.light_object_0_data, offset)

    def load_scene_checkpoint(self, path: Path):
        if not path.exists():
            logging.warning(f"Scene checkpoint not found at {path}, using current parameters.")
            return
        data = torch.load(str(path), map_location="cpu", weights_only=False)
        for k, v in data["scene_params"].items():
            try:
                self.latent.params[k] = v
            except KeyError:
                logging.warning(f"Key `{k}` not found in latent scene params.")
                continue
        self.latent.params.update()
