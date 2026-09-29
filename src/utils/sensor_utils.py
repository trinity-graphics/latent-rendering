from pathlib import Path

import drjit as dr
import mitsuba as mi
import numpy as np


def get_rotated_transform(
    to_world: mi.ScalarTransform4f, phi: float = 0, theta: float = 0
):
    """
    Rotates a given to_world around the center of a sphere by a given phi and theta.
    """
    sensor_distance = dr.norm(to_world.translation())
    new_origin = mi.ScalarPoint3f([0, 0, 0])
    new_target = mi.ScalarPoint3f([0, 0, sensor_distance])
    up = mi.ScalarVector3f([0, 1, 0])
    rotation = (
        mi.ScalarTransform4f()
        .translate(new_target)
        .rotate(axis=[1, 0, 0], angle=theta)
        .rotate(axis=[0, 1, 0], angle=phi)
        .translate(-new_target)
    )
    new_origin = to_world @ rotation @ new_origin
    new_target = to_world @ rotation @ new_target
    return mi.ScalarTransform4f().look_at(new_origin, new_target, up)

def get_rotated_translated_transform(
    to_world: mi.ScalarTransform4f, phi: float = 0, theta: float = 0, offset: mi.ScalarVector3f = mi.ScalarVector3f(0, 0, 0)
):
    """
    Rotates a given to_world around the center of a sphere by a given phi and theta.
    """
    sensor_distance = dr.norm(to_world.translation())
    new_origin = mi.ScalarPoint3f([0, 0, 0])
    new_target = mi.ScalarPoint3f([0, 0, sensor_distance])
    up = mi.ScalarVector3f([0, 1, 0])
    rotation = (
        mi.ScalarTransform4f()
        .translate(new_target)
        .rotate(axis=[1, 0, 0], angle=theta)
        .rotate(axis=[0, 1, 0], angle=phi)
        .translate(-new_target)
    )
    new_origin = to_world @ rotation @ new_origin
    new_target = to_world @ rotation @ new_target
    transform = mi.ScalarTransform4f().look_at(new_origin, new_target, up).translate(offset)
    # print(transform)
    return transform

def load_sensor_like(
    sensor: mi.Sensor,
    to_world: mi.ScalarTransform4f = None,
    phi: float = 0,
    theta: float = 0,
    return_world_transform=False,
    sampler=None
):
    r"""
    Loads a copy of a sensor and places it on the surface of the sphere defined by the original sensor's look-at target (sphere center) and position (radius).
    Offsets the new sensor by a given azimuth and elevation.

    Args:
        sensor (`mi.Sensor`):
            The sensor to copy, which will provide references to its sampler and film.
        to_world (`mi.ScalarTransform4f`, *optional*, defaults to `None`):
            The world transform of the sensor.  While this can be extracted from the sensor directly
            if `to_world` = `None`, there is overhead involved, so it is recommended to extract it and save it
            for multiple calls to this function.
        phi (`float`, *optional*, defaults to `0`):
            The azimuth offset (in degrees) for the new camera's position on the sphere surface.
        theta (`float`, *optional*, defaults to `0`):
            The elevation offset (in degrees) for the new camera's position on the sphere surface.
        return_world_transform (`bool`, *optional*, defaults to `False`):
            Whether or not to directly return a to_world matrix rather than instantiating a new sensor.
            Useful in cases where one wants to update an existing camera's world transform.
    """

    if to_world is None:
        to_world = mi.ScalarTransform4f(
            sensor.world_transform().matrix.numpy().squeeze(-1)
        )

    # Get new sensor's world transform
    new_sensor_to_world = get_rotated_transform(to_world, phi, theta)

    # Optionally, just return the new transform
    if return_world_transform:
        return new_sensor_to_world

    # List of Samplers
    sampler_types = ['independent', 'multijitter', 'stratified', 'orthogonal']
    if sampler in sampler_types:
        sensor_sampler = mi.load_dict({
            'type': sampler
        })
    else:
        sensor_sampler = sensor.sampler()

    # Otherwise, return a new camera instance
    sensor_type = "perspective"
    sensor_film = sensor.film()
    sensor_x_fov = mi.traverse(sensor)["x_fov"][0]
    sensor_fov_axis = "x"
    sensor_near_clip = sensor.near_clip()
    sensor_far_clip = sensor.far_clip()

    sensor_dict = {
        "type": sensor_type,
        "fov": sensor_x_fov,
        "fov_axis": sensor_fov_axis,
        "near_clip": sensor_near_clip,
        "far_clip": sensor_far_clip,
        "to_world": new_sensor_to_world,
        "sampler": sensor_sampler,
        "film": sensor_film,
    }

    return mi.load_dict(sensor_dict)


def set_rgb_film(sensor, res: int = 512, rfilter_type = "box"):
    hdr_film = mi.load_dict(
        {
            "type": "hdrfilm",
            "width": res,
            "height": res,
            "rfilter": {"type": rfilter_type},
            "pixel_format": "rgb",
            "component_format": "float32",
        }
    )
    sensor.m_film = hdr_film

def set_latent_film(sensor, res: int = 64):
    latent_film = mi.load_dict(
        {
            "type": "latfilm",
            "width": res,
            "height": res,
            "rfilter": {"type": "box"},
            "component_format": "float32",
        }
    )
    sensor.m_film = latent_film

def gaussian_dist(x, mean = 0, std = 1):
    if std == 0:
        y = np.ones_like(x) / (x.shape[0])
        return y
    
    scale = 1 / (std * np.sqrt(2 * np.pi))
    y = -0.5 * np.square((x - mean) / std)
    y = scale * np.exp(y)
    return y

def square_dist(x, start = 0, width = 5):
    y_zeros = np.zeros_like(x)
    y_ones = np.ones_like(x)
    y = np.where((x >= start) & (x <= start + width), y_ones, y_zeros)
    return y

def set_spectral_film(sensor, res: int = 512, channels: int = 4, irregular=False, generate_spd=False, generate_spd_gap=1, resolution = 256):
    spec_dict = {
        "type": "specfilm",
        "width": res,
        "height": res,
        "rfilter": {"type": "box"},
    }

    spd_dir = Path("spds")
    spd_files = list(spd_dir.glob("*.spd"))
    num_spds = max([int(f.stem) for f in spd_files]) + 1

    if num_spds < channels:
        print(
            f"WARNING: There are fewer available SPDs ({num_spds}) than there are channels ({channels}) in the spectral film.  Will apply SPDs to channels in a round-robin fashion."
        )

    for ch in range(channels):
        spd_file = spd_dir / f"{ch % num_spds}.spd"
        if irregular:
            wavelengths = []
            values = []

            with open(spd_file, "r") as spd:
                for line in spd:
                    line = line.strip()
                    if line == "" or line.startswith("#"):
                        continue
                    wl, val = line.split()
                    wavelengths.append(str(int(float(wl))))
                    values.append(f"{float(val):.10f}")

            spec_dict[f"band{ch}"] = {
                "type": "irregular",
                "wavelengths": ",".join(wavelengths),
                "values": ",".join(values),
            }
        elif generate_spd:
            wavelengths = []
            values = []
            
            x = np.linspace(0, resolution, resolution)
            wl = (x / resolution) * 350 + 350
            width = resolution // channels
            val = square_dist(x, ch * width, width - generate_spd_gap)
            for idx in range(resolution):
                wavelengths.append(str(int(wl[idx])))
                values.append(f"{float(val[idx]):.10f}")
                
            spec_dict[f"band{ch}"] = {
                "type": "irregular",
                "wavelengths": ",".join(wavelengths),
                "values": ",".join(values),
            }
        else:
            spec_dict[f"band{ch}"] = {"type": "spectrum", "filename": str(spd_file)}

    mi_var = mi.variant()
    if "spectral" in mi_var:
        spec_film = mi.load_dict(spec_dict)
        sensor.m_film = spec_film
    elif "latent" in mi_var:
        set_latent_film(sensor, res)
    else:
        raise RuntimeError(f"Invalid variant `{mi_var}` set, must be one of [...spectral, ...latent4, ...latent16].")
    
