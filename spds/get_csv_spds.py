# This file splits the CSV from https://haraldbrendel.com/ledspd.html
# into multiple SPD files, used for Mitsuba 3's spectral rendering mode.
import csv

import numpy as np

num_spds = 64

in_csv = "led_spd_350_700.csv"
source_url = "https://haraldbrendel.com/ledspd.html"
extract_rows = [int(i) for i in np.linspace(0, 28, num=num_spds, dtype=np.int64)]

with open(in_csv, 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)

wavelengths = [float(w) for w in header]

for i, row_index in enumerate(extract_rows):
    if row_index >= len(rows):
        raise IndexError(f"Row index {row_index} out of bounds.")
    
    row = rows[row_index]

    with open(f"{i}.spd", 'w') as f:
        f.write(f"# Source: {source_url}\n")
        f.write(f"# Row Index: {row_index}\n")

        for wl, val in zip(wavelengths, row):
            value = float(val)
            f.write(f"{wl:.2f} {value:.10f}\n")