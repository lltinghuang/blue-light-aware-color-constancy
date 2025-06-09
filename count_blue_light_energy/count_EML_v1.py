import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# Step 1: 我們的光譜 CSV（不是等間距）
our_spd = pd.read_csv("../test_spd_result/test_0.9_spd.csv") 
# our_spd = our_spd.sort_values('wavelength')

# Step 2: 讀CIE α-opic.csv，只取 Melanopic 欄
cie_all = pd.read_csv("CIE_a-opic_action_spectra.csv", header=None) # CIE official melanopic action spectrum(光敏曲線)
cie_all.columns = ['wavelength', 'Scone', 'Mcone', 'Lcone', 'Melanopic', 'Rhodopic']
cie_data = cie_all[['wavelength', 'Melanopic']].dropna()

# Step 3: 把原始光譜內插為 CIE 的波長（380–780 nm)
interp_func = interp1d(our_spd['wavelength'], our_spd['intensity'], kind='linear',
                       fill_value=0, bounds_error=False)

cie_data['intensity'] = interp_func(cie_data['wavelength'])

# Step 4: 計算積分與 EML
delta_lambda = 1  # 每 nm 一筆
cie_data['product'] = cie_data['intensity'] * cie_data['Melanopic']
EML_relative = 72983.25 * cie_data['product'].sum() * delta_lambda

print(f"EML = {EML_relative:.4f}（單位為 lux 相對值，未校正）")
