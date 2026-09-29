from __future__ import annotations

import math
import os
import subprocess
from collections.abc import Sequence
from itertools import product

import drjit as dr
import flip_evaluator as flip
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
import torch
from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity,
)
from skimage.util import montage
from tqdm.auto import tqdm

from utils.losses import lpipsLoss

LUMINANCE_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

LAT2RGB_T = torch.tensor(
    [
        [0.298, 0.207, 0.208],
        [0.187, 0.286, 0.173],
        [-0.158, 0.189, 0.264],
        [-0.184, -0.271, -0.473],
    ]
)  # .to(device=device)

RGB2LAT_T = torch.tensor(
    [
        [1.698, -0.283, -2.552, -0.781],
        [-0.030, 4.914, 2.232, 3.030],
        [-0.057, -3.048, 0.045, -3.229],
    ]
)  # .to(device=device)


def to_image_tensor(x, device=None, dtype=torch.float32) -> torch.Tensor:
    if device is None:
        device = torch.device("cuda" if "cuda" in mi.variant() else "cpu")

    # Normalize to PyTorch tensor
    if isinstance(x, torch.Tensor):
        t = x
    elif isinstance(x, np.ndarray):
        t = torch.from_numpy(np.ascontiguousarray(x))
    elif isinstance(x, mi.TensorXf):
        t = x.torch()
    else:
        raise TypeError(f"Unsupported type: {type(x)}")

    t = t.to(device=device, dtype=dtype, non_blocking=True)

    # Validate shape
    if t.ndim != 4 or t.shape[0] != 1:
        raise ValueError(f"Expected shape (1, C, H, W), got {tuple(t.shape)}. ")

    return t


def pt2mi(image: torch.Tensor) -> mi.TensorXf:
    return mi.TensorXf(image.squeeze(0).permute(1, 2, 0).contiguous())


def mi2pt(image: mi.TensorXf) -> torch.Tensor:
    return torch.from_dlpack(image).permute(2, 0, 1).unsqueeze(0)


def invert_value(image, coord_x, coord_y, ch=0):
    image = np.array(image)
    image[coord_y][coord_x][ch] = -image[coord_y][coord_x][ch]
    return mi.TensorXf(image)


def invert_superpixel(image, coord_x, coord_y):
    image = np.array(image)
    image[coord_y][coord_x] = -image[coord_y][coord_x]
    return mi.TensorXf(image)


def dr_sigmoid(x):
    return 1 / (1 + dr.exp(-x))


def dr_logit(x):
    return dr.log(x / (1 - x))


def scale(x, min, max):
    return min + x * (max - min)


def normalize(x, min, max):
    return (x - min) / (max - min)


def reinhard(x):
    # Reinhard tone mapping
    return x / (1 + x)


def dr_gamma(x):
    # sRGB gamma curve approximation
    return dr.power(x, 1.0 / 2.2)


def srgb_to_linear(x):
    a = 0.055
    out = np.empty_like(x, dtype=float)
    mask = x <= 0.04045
    out[mask] = x[mask] / 12.92
    out[~mask] = ((x[~mask] + a) / (1 + a)) ** 2.4
    return out


def lat2rgb(lat):
    return (lat.view(-1, 4) @ LAT2RGB_T).view(64, 64, 3)


def rgb2lat(rgb):
    return (rgb.view(-1, 3) @ RGB2LAT_T).view(512, 512, 4)


def reinhard_tm(img):
    abs_val = torch.abs(img)
    res = abs_val / (1 + abs_val)
    return torch.pow(res, 1 / 2.2)


def clip_tm(img):
    res = torch.clip(img, 0, 1)
    return torch.pow(res, 1 / 2.2)


def reinhard_luminance(img):
    img = np.asarray(img, dtype=np.float32)
    return img / (1 + img @ LUMINANCE_WEIGHTS)[..., None]


def max_spp_per_pass(film):
    width, height = film.crop_size()
    if film.sample_border():
        border = 2 * film.rfilter().border_size()
        width, height = width + border, height + border
    return (2**32 - 1) // (width * height)


def render_converged(scene, total_spp, chunk_spp=None, sensor=0, seed=0, out_stem=None):
    if isinstance(sensor, int):
        sensor = scene.sensors()[sensor]
    film, integrator = sensor.film(), scene.integrator()
    chunk_spp = min(chunk_spp or total_spp, max_spp_per_pass(film), total_spp)

    width, height = film.crop_size()
    weight_ch = film.base_channels_count()
    accum = np.zeros((height, width, weight_ch + 1), dtype=np.float64)
    remaining = total_spp

    with dr.suspend_grad(), tqdm(total=total_spp, unit="spp") as progress:
        while remaining > 0:
            spp = min(chunk_spp, remaining)
            integrator.render(scene, sensor, seed=seed, spp=spp, develop=False)
            accum += np.asarray(film.develop(raw=True))[..., :weight_ch + 1]
            seed, remaining = seed + 1, remaining - spp
            progress.update(spp)

    colour, weight = accum[..., :3], accum[..., weight_ch:]
    rgb = np.divide(colour, weight, out=np.zeros_like(colour), where=weight > 0)
    image = mi.TensorXf(np.ascontiguousarray(rgb, dtype=np.float32))

    if out_stem is not None:
        mi.util.write_bitmap(f"{out_stem}.exr", image, write_async=False)
        mi.util.write_bitmap(f"{out_stem}.png", reinhard_luminance(image), write_async=False)
    return image


def get_tiled_like(
    big: mi.TensorXf | Sequence,
    small: mi.TensorXf | Sequence,
    mode: str = "r",
    backend="np",
):
    """
    Method for reading/writing multiple low-resolution channels into/out of a higher resolution in a tiled manner.
    When `mode`=`'w'`, the channels of `small` are tiled into `big`'s resolution, and written to the minimum number of channels possible.
    When `mode`=`'r'`, the tiles in `big` will be extracted to `small`'s shape. (This assumes `big` contains only the tiled image channels,
    which have been separated from their associated full-resolution channels with `mi.Bitmap().split()`.)
    """
    if mode not in ["w", "r"]:
        raise ValueError(
            "Mode must be 'w' for tiling image to host resolution, or 'r' for extracting tiled channels from a host image!"
        )
    if backend not in ["dr", "np"]:
        raise ValueError(
            "Backend must be one of `dr` (for drjit arrays), or `np` (for numpy arrays)."
        )

    if isinstance(big, mi.TensorXf) or isinstance(big, np.ndarray):
        H, W, C = big.shape
    elif isinstance(big, Sequence):
        H, W, C = big
    else:
        raise TypeError(f"Invalid parameter type {type(big)} for `big`.")

    if isinstance(small, mi.TensorXf) or isinstance(small, np.ndarray):
        h, w, c = small.shape
    elif isinstance(small, Sequence):
        h, w, c = small
    else:
        raise TypeError(f"Invalid parameter type {type(small)} for `small`.")

    th = H // h  # How many tiles fit into the height of the image
    tw = W // w  # How many tiles fit into the width of the image
    # How many host-resolution channels will be needed to fit all image-resolution channels ('w')
    # OR how many host-resolution channels are available. ('r')
    tc = math.ceil(c / (th * tw)) if mode == "w" else C
    ti = list(product(range(0, H, h), range(0, W, w), range(tc)))

    if backend == "dr":
        make_arr = dr.zeros
        arr_type = mi.TensorXf
    elif backend == "np":
        make_arr = np.zeros
        arr_type = small.dtype if mode == "w" else big.dtype
    arr_shape = (H, W, tc) if mode == "w" else (h, w, c)

    out = make_arr(dtype=arr_type, shape=arr_shape)

    for ch in range(c):
        y, x, z = ti[ch]

        if mode == "w":
            inslc = small[:, :, ch]
            if backend == "dr":
                inslc = inslc.array
            out[y : y + h, x : x + w, z] = inslc
        elif mode == "r":
            inslc = big[y : y + h, x : x + w, z]
            if backend == "dr":
                inslc = inslc.array
            out[..., ch] = inslc

    return out


##
##  LATENT-SPACE EXR
##
def save_latent_exr(latent, name="latent"):
    """
    Saves latent values to an EXR file losslessly.
    """
    latent = to_image_tensor(latent)

    if torch.any(torch.isnan(latent)):
        print(f"LATENT EMERGENCY! NaN VALUES FOUND IN {name}")

    latent = pt2mi(latent)

    bmp = mi.Bitmap(
        latent,
        pixel_format=mi.Bitmap.PixelFormat.MultiChannel,
        # channel_names = [f"ch{c:02d}" for c in range(latent.shape[-1])]
    )
    bmp.set_srgb_gamma(False)
    bmp.write_async(f"{name}.exr")


def load_latent_exr(name="latent"):
    bmp = mi.Bitmap(f"{name}.exr")
    bmp.set_srgb_gamma(False)
    return mi2pt(mi.TensorXf(bmp))


##
##  LATENT-SPACE PNG
##
def to_posneg(image: np.ndarray):
    image = 2 * torch.sigmoid(image) - 1

    pos = image.clip(min=0)
    neg = (-image).clip(min=0)
    zeros = torch.zeros_like(image)

    # Out shape is BC3HW
    image = torch.stack([neg, pos, zeros], dim=2)
    return image


def save_latent_png(latent, name="latent"):
    latent = to_image_tensor(latent)

    # Apply pos/neg tonemap and montage latent channels
    latent = to_posneg(latent)
    latent = np.asarray(latent.squeeze(0).detach().cpu())
    latent = montage(latent, fill=(0, 0, 0), rescale_intensity=False, channel_axis=1)
    # The resultant array is in (H, W, C) order.

    # Write bitmap
    bmp = mi.Bitmap(latent, mi.Bitmap.PixelFormat.RGB)
    bmp.set_srgb_gamma(False)
    bmp = bmp.convert(component_format=mi.Struct.Type.UInt8, srgb_gamma=False)
    bmp.write_async(f"{name}.png")


def load_latent_png(name):
    # By default, images loaded from PNG will be treated as gamma corrected.
    bmp = mi.Bitmap(f"{name}.png")
    bmp.set_srgb_gamma(False)
    bmp = bmp.convert(component_format=mi.Struct.Type.Float32, srgb_gamma=False)
    return mi2pt(mi.TensorXf(bmp))


##
##  IMAGE-SPACE EXR
##
def save_image_exr(image, name="image"):
    image = to_image_tensor(image)
    image = pt2mi(image)
    bmp = mi.Bitmap(image, mi.Bitmap.PixelFormat.RGB)
    bmp.set_srgb_gamma(False)
    bmp.write_async(f"{name}.exr")


def load_image_exr(name="image"):
    bmp = mi.Bitmap(f"{name}.exr")
    bmp.set_srgb_gamma(False)
    return mi2pt(mi.TensorXf(bmp))


##
##  IMAGE-SPACE PNG
##
def save_image_png(image, name="image"):
    image = pt2mi(image)

    bmp = mi.Bitmap(image)
    bmp.set_srgb_gamma(False)

    if bmp.channel_count() == 3:
        pf = mi.Bitmap.PixelFormat.RGB
    elif bmp.channel_count() == 1:
        pf = mi.Bitmap.PixelFormat.Y

    bmp = bmp.convert(pf, mi.Struct.Type.UInt8, srgb_gamma=False)
    bmp.write_async(f"{name}.png")


def load_image_png(name="image"):
    bmp = mi.Bitmap(f"{name}.png")
    bmp.set_srgb_gamma(False)
    bmp = bmp.convert(component_format=mi.Struct.Type.Float32, srgb_gamma=False)
    return mi2pt(mi.TensorXf(bmp))


##
##
##

# def mean_flip_error(true, test, is_latent=False):
#     if is_latent:
#         true = true.squeeze(0).permute(1, 2, 0)
#         test = test.squeeze(0).permute(1, 2, 0)
#         true_pos = torch.sigmoid(true)
#         test_pos = torch.sigmoid(test)
#         true_np = np.asarray(true_pos.detach().cpu())
#         test_np = np.asarray(test_pos.detach().cpu())
#         error_map, error, _ = flip.evaluate(true_np, test_np, "HDR")
#     else:
#         true = true.squeeze(0).permute(1, 2, 0)
#         test = test.squeeze(0).permute(1, 2, 0)
#         true_np = np.asarray(true.detach().cpu())
#         test_np = np.asarray(test.detach().cpu())
#         error_map, error, _ = flip.evaluate(true_np, test_np, "LDR")
#     error_map = torch.as_tensor(error_map).permute(2, 0, 1).unsqueeze(0)
#     return error_map, error


def flip_error(true, test):
    true = true.transpose(1, 2, 0)
    test = test.transpose(1, 2, 0)
    error_map, error, _ = flip.evaluate(true, test, "LDR")
    return error_map, error


def compute_image_metrics(true, test, is_latent=False):
    with torch.no_grad():
        metrics = {}

        if true.shape[0] != 1 or test.shape[0] != 1:
            raise ValueError("Image metrics computation requires batch size 1 inputs.")
        true = true.squeeze(0)
        test = test.squeeze(0)

        if isinstance(true, torch.Tensor):
            true = np.asarray(true.detach().cpu())
        if isinstance(test, torch.Tensor):
            test = np.asarray(test.detach().cpu())

        # get data ranges if latent!
        if is_latent:
            # data_range = 10     # fixed [-5, 5] range
            data_range = np.max(true) - np.min(true)
        else:
            data_range = 1

        metrics["mse"] = mean_squared_error(true, test)
        metrics["mse_lin"] = mean_squared_error(
            srgb_to_linear(true), srgb_to_linear(test)
        )
        metrics["psnr"] = peak_signal_noise_ratio(true, test, data_range=data_range)
        metrics["ssim"] = structural_similarity(
            true,
            test,
            data_range=data_range,
            channel_axis=0 if true.ndim > 2 else None,
        )
        metrics["mean"] = np.mean(test)
        metrics["var"] = np.var(test)
        if not is_latent:
            metrics["lpips"] = np.float32(lpipsLoss(true, test))
            flip_emap, flip_err = flip_error(true, test)
            metrics["flip"] = flip_err
            return metrics, torch.tensor(flip_emap).permute(2, 0, 1).unsqueeze(0)

        return metrics, None


def print_image_stats(image: torch.Tensor, debug_name: str):
    print(f"\n### [DEBUG] {debug_name} ###")
    flat = image.reshape(-1)
    minv = flat.min()
    maxv = flat.max()
    mean = flat.mean()
    var = flat.var()
    print(f"Range: [{minv.item()}, {maxv.item()}]")
    print(f"µ: {mean.item()}")
    print(f"σ^2: {var.item()}")


def decode_and_write(mi_vae, latent, path):
    image = mi_vae.decode(latent)
    save_image_png(image, path)


def visualize_latents(latent, out_dir, prefix_str="", latent_min=-18, latent_max=18):
    # Perform scaling if needed
    if dr.max(latent) > 1.0:
        # latent = sigmoid(latent)
        latent = normalize(latent, latent_min, latent_max)

    latent_np = latent.numpy()

    # Output individual channels
    # [mi.Bitmap(latent_np[:,:,c]).convert(mi.Bitmap.PixelFormat.Y, mi.Struct.Type.UInt8).write(f"{out_dir}/{prefix_str}{c}.png") for c in range(latent_np.shape[-1])]

    # Output collage image
    top = np.concatenate([latent_np[:, :, 0], latent_np[:, :, 1]], axis=1)
    bottom = np.concatenate([latent_np[:, :, 2], latent_np[:, :, 3]], axis=1)
    collage = np.concatenate([top, bottom], axis=0)
    mi.Bitmap(collage).convert(mi.Bitmap.PixelFormat.Y, mi.Struct.Type.UInt8).write(
        f"{out_dir}/{prefix_str}-latent.png"
    )


def visualize_value_distribution(image, out_dir, prefix_str=""):
    # if dr.max(image) > 1.0:
    #     image = normalize(image, min, max)

    data = np.array(dr.ravel(image))
    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=50, density=True, alpha=0.6, color="skyblue", edgecolor="black")
    plt.title("Histogram of Values")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.grid(True)
    plt.savefig(f"{out_dir}/{prefix_str}-distribution.png")


def write_gif(in_imgs, out_gif, framerate):
    palette_path = "palette.png"
    palette_cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(framerate),
        "-i",
        in_imgs,
        "-vf",
        "palettegen",
        palette_path,
    ]
    gif_cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(framerate),
        "-i",
        in_imgs,  # Input images, example "move-light-%d.png"
        "-i",
        palette_path,
        "-filter_complex",
        "[0:v]split[main][copy];[copy]reverse[rev];[main][rev]concat=n=2:v=1[x];[x][1:v]paletteuse=dither=none[out]",  # "Boomerang" effect
        "-map",
        "[out]",
        "-loop",
        "0",
        f"{out_gif}.gif",  # Output gif name
    ]
    try:
        subprocess.run(
            palette_cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            gif_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        os.remove(palette_path)
        print(f"GIF created successfully: {out_gif}.gif")
    except subprocess.CalledProcessError as e:
        print("ffmpeg failed with error:")
        print(e)


def write_mp4(in_imgs, out_mp4, framerate, upscale=None):
    if upscale is not None:
        scale_filter = f"scale=iw*{upscale}:ih*{upscale}:flags=neighbor,"
    else:
        scale_filter = ""

    filter_complex = (
        f"[0:v]{scale_filter}split[main][copy];"
        "[copy]reverse[rev];"
        "[main][rev]concat=n=2:v=1[x]"
    )

    mp4_cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(framerate),
        "-i",
        in_imgs,  # e.g. "output_%04d.png" for zero-padded files
        "-filter_complex",
        filter_complex,
        "-map",
        "[x]",
        "-c:v",
        "libx264",
        "-preset",
        "veryslow",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        f"{out_mp4}.mp4",
    ]

    try:
        subprocess.run(
            mp4_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        print(f"MP4 created successfully: {out_mp4}.mp4")
    except subprocess.CalledProcessError as e:
        print("ffmpeg failed with error:")
        print(e)


def upsample(in_img, out_img, scale=8):
    upsamp_command = [
        "ffmpeg",
        "-y",
        "-i",
        in_img,
        "-vf",
        f"scale=iw*{scale}:ih*{scale}:flags=neighbor",
        "-frames:v",
        "1",
        "-update",
        "1",
        out_img,
    ]

    try:
        subprocess.run(
            upsamp_command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as e:
        print("ffmpeg failed with error:")
        print(e)
