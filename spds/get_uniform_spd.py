# Generate an SPD file with uniform power from wavelengths 350 to 700.

wavelengths = [float(i) for i in range(350, 700, 2)]
values = [1 / len(wavelengths)] * len(wavelengths)

source_url = "https://haraldbrendel.com/ledspd.html"

with open("uniform.spd", 'w') as f:
    f.write(f"# Uniform SPD corresponding to wavelengths used in {source_url}")

    f.writelines(f"{wl:.2f} {val:.10f}\n" for wl, val in zip(wavelengths, values))
