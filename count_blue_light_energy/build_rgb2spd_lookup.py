import os
from collections import defaultdict
from glob import glob

import numpy as np

gamma = 2.24
base_dir = "./exp_data/hauwei_mt/temperature_change_max_light_expose_time_1000"
cache_dir = "./rgb2spd_lookup"
os.makedirs(cache_dir, exist_ok=True)

def read_spectrum_csv(file_path):
    wavelengths = []
    intensities = []
    with open(file_path, 'r', encoding='utf-8') as f:
        begin = False
        for line in f:
            if not begin and "BEGIN" in line:
                begin = True
                continue
            if not begin:
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

def save_spd_cache_from_group(mode, temp, group):
    try:
        wl, spd_R = read_spectrum_csv(group['red'])
        _, spd_G = read_spectrum_csv(group['green'])
        _, spd_B = read_spectrum_csv(group['blue'])
    except KeyError as e:
        print(f"[SKIP] Missing {e.args[0]} for {mode}_{temp}k")
        return

    # Gamma decoding
    spd_R = np.power(np.maximum(spd_R, 0), 1 / gamma)
    spd_G = np.power(np.maximum(spd_G, 0), 1 / gamma)
    spd_B = np.power(np.maximum(spd_B, 0), 1 / gamma)

    # === Build cache directory ===
    cache_dir_64 = os.path.join(cache_dir, "float64")
    cache_dir_32 = os.path.join(cache_dir, "float32")
    os.makedirs(cache_dir_64, exist_ok=True)
    os.makedirs(cache_dir_32, exist_ok=True)

    # === for float64 ===
    out64 = os.path.join(cache_dir_64, f"{mode}_{temp}K.npz")
    np.savez(out64, wl=wl, spd_R=spd_R, spd_G=spd_G, spd_B=spd_B)
    print(f"[OK] Saved float64 → {out64}")

    # === for float32 ===
    out32 = os.path.join(cache_dir_32, f"{mode}_{temp}K.npz")
    np.savez(out32,
             wl=wl.astype(np.float32),
             spd_R=spd_R.astype(np.float32),
             spd_G=spd_G.astype(np.float32),
             spd_B=spd_B.astype(np.float32))
    print(f"[OK] Saved float32 → {out32}")

# Group all files by (mode, temp)
file_groups = defaultdict(dict)

for file_path in glob(os.path.join(base_dir, "*.csv")):
    fname = os.path.basename(file_path)
    # ex: hauwei_2700k_red.csv
    parts = fname.replace(".csv", "").split("_")
    if len(parts) < 3:
        continue
    mode, temp, color = parts[0], parts[1], parts[2]
    temp = temp.lower().replace("k", "")
    file_groups[(mode, temp)][color.lower()] = file_path

# Save SPD npz for each group
for (mode, temp), group in file_groups.items():
    save_spd_cache_from_group(mode, temp, group)
