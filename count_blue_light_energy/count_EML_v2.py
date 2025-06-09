import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# CLI Argument Parser
parser = argparse.ArgumentParser(description="Compute EML from SPD and optionally save image.")
parser.add_argument('--spd_path', type=str, required=True, help='Path to SPD CSV (wavelength,intensity)')
parser.add_argument('--image', type=str, help='Path to save image (e.g. eml_plot.png)', default=None)
args = parser.parse_args()

# Step 1: 讀光譜、CIE Melanopic、Photopic V(λ)
if not os.path.exists(args.spd_path):
    raise FileNotFoundError(f"找不到 SPD 檔案：{args.spd_path}")
our_spd = pd.read_csv(args.spd_path)

# 讀取CIE melanopic sensitivity 的 'Melanopic' (alpha-opic curve)
cie_all = pd.read_csv("./CIE_a-opic_action_spectra.csv", header=None)
cie_all.columns = ['wavelength', 'Scone', 'Mcone', 'Lcone', 'Melanopic', 'Rhodopic']
melanopic = cie_all[['wavelength', 'Melanopic']].dropna()

# 讀取 Photopic V(λ) 曲線：人眼對不同波長亮度感知靈敏度的曲線（CIE 1931 視覺靈敏度函數）
cie_v = pd.read_csv("./CIE_sle_photopic.csv", header=None)
cie_v.columns = ['wavelength', 'V_lambda']

# Step 2: SPD 線性內插至 1nm（對齊 CIE 波長）
interp_func = interp1d(our_spd['wavelength'], our_spd['intensity'], kind='linear',
                       fill_value=0, bounds_error=False)
wavelengths = melanopic['wavelength']
intensity_interp = interp_func(wavelengths)

# Step 3: 合併三個 DataFrame（melanopic、photopic、interpolated SPD）
df = pd.DataFrame({
    'wavelength': wavelengths,
    'intensity': intensity_interp
})
df = df.merge(melanopic, on='wavelength')
df = df.merge(cie_v, on='wavelength')

# Step 4: 計算 Photopic、Melanopic、EML
photopic_lux = np.sum(df['intensity'] * df['V_lambda']) * 683
melanopic_lux = np.sum(df['intensity'] * df['Melanopic']) * 72983.25
melanopic_ratio = melanopic_lux / photopic_lux if photopic_lux > 0 else 0
EML = photopic_lux * melanopic_ratio

# Step 5: 結果
print(f"\n【計算結果】")
print(f"SPD 檔案：{args.spd_path}")
print(f"Photopic Illuminance = {photopic_lux:.2e}（相對 lux）")
print(f"Melanopic Ratio       = {melanopic_ratio:.4f}")
print(f"EML                   = {EML:.2e}（相對 lux）\n")

# === Step 6: 畫圖 ===
df['V_weighted'] = df['intensity'] * df['V_lambda']
df['mel_weighted'] = df['intensity'] * df['Melanopic']

plt.figure(figsize=(10, 6))
plt.plot(df['wavelength'], df['intensity'], label='Original SPD (counts)', linestyle='--', alpha=0.6)
plt.plot(df['wavelength'], df['V_weighted'], label='SPD × V(λ)', linewidth=2)
plt.plot(df['wavelength'], df['mel_weighted'], label='SPD × Melanopic', linewidth=2)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Relative Intensity / Weighting')
plt.title('Spectral Distribution and α-opic Weighting Curves')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Step 7: 儲存或顯示
if args.image:
    plt.savefig(args.image, dpi=300)
else:
    plt.show()
