import csv
import os

import numpy as np
import pandas as pd

# Settings
gamma = 2.24
green_band = (510, 570)
data_dir = "../exp_data/hauwei_mt/Different_rgb/Fix_channel_different_intensity"
files = [
    ("0_43_0.csv", (0, 43, 0)),
    ("0_85_0.csv", (0, 85, 0)),
    ("0_128_0.csv", (0, 128, 0)),
    ("0_170_0.csv", (0, 170, 0)),
    ("0_213_0.csv", (0, 213, 0)),
    ("0_255_0.csv", (0, 255, 0)),
]

# Read spectrum CSV files
def read_spectrum_csv(file_path):
    wavelengths = []
    intensities = []
    with open(file_path, 'r', encoding='utf-8') as f:
        begin = False
        for line in f:
            if not begin:
                if "BEGIN" in line:
                    begin = True
                continue
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            try:
                wl = float(parts[0].strip())
                val = float(parts[1].strip())
                wavelengths.append(wl)
                intensities.append(val)
            except ValueError:
                continue
    return np.array(wavelengths), np.array(intensities)

# Read base spectrum
base_rgb_tuple, base_path = files[0][1], os.path.join(data_dir, files[0][0])
wl_base, spd_base = read_spectrum_csv(base_path)
spd_base = np.maximum(spd_base, 0)
green_mask = (wl_base >= green_band[0]) & (wl_base <= green_band[1])
base_rgb_val = base_rgb_tuple[1]

rows_full = []
rows_green = []

for fname, target_rgb_tuple in files[1:]:
    # Read target spectrum
    path = os.path.join(data_dir, fname)
    wl, spd = read_spectrum_csv(path)
    spd = np.maximum(spd, 0)

    # Calculate expected ratio
    expected = target_rgb_tuple[1] / base_rgb_val
    base_rgb_str = f"{base_rgb_tuple[0]} {base_rgb_tuple[1]} {base_rgb_tuple[2]}"
    target_rgb_str = f"{target_rgb_tuple[0]} {target_rgb_tuple[1]} {target_rgb_tuple[2]}"

    # Full spectrum
    full_ratio = spd / (spd_base + 1e-6)
    avg_full = np.mean(full_ratio)
    maxdev_full = np.max(np.abs(full_ratio - expected))
    rows_full.append([base_rgb_str, target_rgb_str, expected, avg_full, maxdev_full])

    # Green band (510–570nm)
    green_ratio = spd[green_mask] / (spd_base[green_mask] + 1e-6)
    avg_green = np.mean(green_ratio)
    maxdev_green = np.max(np.abs(green_ratio - expected))
    rows_green.append([base_rgb_str, target_rgb_str, expected, avg_green, maxdev_green])

# Create DataFrames for results
df_full = pd.DataFrame(rows_full, columns=[
    "Base RGB", "Target RGB", "Expected Ratio",
    "Avg Ratio (Full)", "Max Dev (Full)"
])
df_green = pd.DataFrame(rows_green, columns=[
    "Base RGB", "Target RGB", "Expected Ratio",
    "Avg Ratio (Green)", "Max Dev (Green)"
])

pd.options.display.float_format = '{:.3f}'.format

# Print the results
print("\n======================= Ratio using FULL spectrum =======================")
print(df_full.to_string(index=False))

print("\n=================== Ratio using GREEN (510–570nm) band ===================")
print(df_green.to_string(index=False))
