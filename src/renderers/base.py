import drjit as dr
import mitsuba as mi

from utils import img_utils


class Renderer:
    def __init__(self):
        self.scene = None
        self.params = None
        self.sensor = None
        self.sensor_params = None
        self.resolution = None
        self.channels = None
        self.spp = None
        self.denoiser = None
        self.debug_out_dir = None
        self.debug_outputs = set()

    @dr.wrap(source="torch", target="drjit")
    def _render(self, seed):
        return mi.render(
            scene=self.scene,
            params=self.params,
            sensor=self.sensor,
            spp=self.spp,
            seed=seed,
        )

    def render(self, seed=0):
        # Render image
        img = self._render(seed)

        # Prepare for use with Pytorch by transforming HWC->BCHW
        img = img.permute(2, 0, 1).unsqueeze(0)
        return img

    def shape(self) -> tuple:
        return (1, self.channels, self.resolution, self.resolution)

    def save_debug_image(
        self, image: mi.TensorXf, name: str, debug: bool | int, save_exr: bool = False
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
                if self.channels > 3:
                    save_fn = img_utils.save_latent_exr
                else:
                    save_fn = img_utils.save_image_exr

                save_fn(image, fname)

            if self.channels > 3:
                save_fn = img_utils.save_latent_png
            else:
                save_fn = img_utils.save_image_png

            save_fn(image, fname)

    def set_spp(self, spp):
        self.spp = spp
