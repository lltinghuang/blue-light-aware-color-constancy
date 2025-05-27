import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Settings
gamma = 2.24
pure_data_dir = "../exp_data/hauwei_mt/Different_rgb"
hybrid_data_dir = "../exp_data/hauwei_mt/Different_rgb/hybrid_color"
fix_data_dir = "../exp_data/hauwei_mt/Different_rgb/Fix_channel_different_intensity"
output_dir = "./plots"
os.makedirs(output_dir, exist_ok=True)

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

# Pure RGB files
pure_rgb_files = {
    "R": ("255_0_0.csv", [255, 0, 0]),
    "G": ("0_255_0.csv", [0, 255, 0]),
    "B": ("0_0_255.csv", [0, 0, 255]),
}

# Hybrid color files
target_files = [
    ("0_128_128.csv", [0, 128, 128]),
    ("128_128_43.csv", [128, 128, 43]),
    ("213_213_85.csv", [213, 213, 85]),
    ("255_255_0.csv", [255, 255, 0]),
    ("85_128_85.csv", [85, 128, 85]),
    ("85_255_128.csv", [85, 255, 128]),
]

# Step 1: Use pure color SPD for gamma correction
spd_dict_lin = {}
for color, (filename, _) in pure_rgb_files.items():
    wl, spd = read_spectrum_csv(os.path.join(pure_data_dir, filename))
    spd = np.maximum(spd, 0)
    spd_lin = np.power(spd, 1 / gamma)
    spd_dict_lin[color] = spd_lin

# Step 2: Predict SPD based on RGB values using gamma correction
def predict_spd_gamma_based(rgb, gamma=gamma):
    r, g, b = rgb
    pred = (
        np.power((spd_dict_lin["R"] * (r / 255)), gamma) +
        np.power((spd_dict_lin["G"] * (g / 255)), gamma) +
        np.power((spd_dict_lin["B"] * (b / 255)), gamma)
    )
    return pred

# Step 3: Predict SPD for hybrid colors
rows = []
for filename, rgb in target_files:
    wl_target, spd_target = read_spectrum_csv(os.path.join(hybrid_data_dir, filename))
    spd_target = np.maximum(spd_target, 0)
    spd_pred = predict_spd_gamma_based(rgb)

    mse = np.mean((spd_pred - spd_target) ** 2)
    corr = np.corrcoef(spd_pred, spd_target)[0, 1]
    rows.append([f"{rgb[0]}_{rgb[1]}_{rgb[2]}", mse, corr])

    plt.figure(figsize=(10, 4))
    plt.plot(wl_target, spd_target, label="Measured", color="black")
    plt.plot(wl_target, spd_pred, label="Predicted", linestyle="--", color="red")
    plt.title(f"RGB = {rgb}")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"SPD_plot_pred_{filename.replace('.csv','')}.png"))
    plt.close()

# Step 4: Predict SPD for Fix_channel_different_intensity foloer
fix_pattern = re.compile(r"(\d{1,3})_(\d{1,3})_(\d{1,3})\.csv")
for fname in os.listdir(fix_data_dir):
    match = fix_pattern.match(fname)
    if not match:
        continue
    rgb = list(map(int, match.groups()))
    wl_target, spd_target = read_spectrum_csv(os.path.join(fix_data_dir, fname))
    spd_target = np.maximum(spd_target, 0)
    spd_pred = predict_spd_gamma_based(rgb)

    mse = np.mean((spd_pred - spd_target) ** 2)
    corr = np.corrcoef(spd_pred, spd_target)[0, 1]
    rows.append([f"{rgb[0]}_{rgb[1]}_{rgb[2]}", mse, corr])

    plt.figure(figsize=(10, 4))
    plt.plot(wl_target, spd_target, label="Measured", color="black")
    plt.plot(wl_target, spd_pred, label="Predicted", linestyle="--", color="red")
    plt.title(f"[FIX Folder] RGB = {rgb}")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"SPD_plot_pred_{fname.replace('.csv','')}.png"))
    plt.close()

# Print the results
df = pd.DataFrame(rows, columns=["RGB", "MSE", "Pearson Corr"])
pd.options.display.float_format = '{:.4f}'.format
print("\n=== Gamma-Free SPD Prediction (Full Dataset) ===")
print(df.to_string(index=False))