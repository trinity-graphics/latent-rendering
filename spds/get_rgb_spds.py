# This file creates 8 SPD files according to the wavelength ranges of visible light.
# To better match previous experiments (and maintain a high parameter count), it
# creates 176 wavelength-power pairs.
# The main difference between this file and get_csv_spds is that the SPDs we create
# here have no overlapping wavelengths.
import numpy as np
from scipy.stats import norm

["r", "o", "y", "g", "c", "b", "v", "uv"]

ranges = {
    "r": (625, 750),
    "o": (590, 625),
    "y": (565, 590),
    "g": (500, 565),
    "c": (485, 500),
    "b": (450, 485),
    "v": (380, 450),
    "uv": (315, 380)
}

def gauss_list(length):
    mean = length // 2
    stddev = length / 6
    x = np.arange(length)
    gaussian = norm.pdf(x, loc=mean, scale=stddev)
    normalized = gaussian / gaussian.sum()
    return normalized.tolist()

for color in ranges:

    wavelengths = list(range(ranges[color][0], ranges[color][1]))
    vals = gauss_list(ranges[color][1] - ranges[color][0])
    
    with open(f"c_{color}.spd", 'w') as f:
        for wl, val in zip(wavelengths, vals):
            value = float(val)
            f.write(f"{wl:.2f} {value:.10f}\n")
    
