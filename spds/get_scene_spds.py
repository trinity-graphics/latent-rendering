#
#
#   This file serves as a one-off conversion between the RGB values of the scene parameters
#   And their 4-channel spectral versions, for easy access.
#   This operation is based on the pseudoinverse of the matrix provided by the following link:
#   https://discuss.huggingface.co/t/decoding-latents-to-rgb-without-upscaling/23204/2
#
#
import code

import mitsuba as mi
import numpy as np

mi.set_variant("llvm_ad_rgb")

RGB2LAT_T = np.array([[1.698, -0.030, -0.057],
                          [-0.283, 4.914, -3.048],
                          [-2.552, 2.232, 0.045],
                          [-0.781, 3.030, -3.229]])

scene = mi.load_file("./scenes/cbox.xml", optimize=False)
params = mi.traverse(scene)

bsdf_params = [p[0] for p in params if ".reflectance.value" in p[0]]
emitter_params = [p[0] for p in params if ".radiance.value" in p[0]]
opt_params = bsdf_params + emitter_params


wavelengths = [400, 500, 600, 700]


# for p in opt_params:
#     filename = f"{p.split(".")[0]}.spd"
#     latent_color = RGB2LAT_T @ np.array(params[p])

#     for 


#   TODO: right now, this is determined by out-of-gamut color range.



code.interact(local=locals())