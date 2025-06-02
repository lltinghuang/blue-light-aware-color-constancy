import os

import cv2
import numpy as np


def extract_avg_spd_from_image(image_path: np.ndarray, temp=6500, resize_max=256):
    """
    Input image & temperature to compute the average spectral power distribution (SPD).

    Parameters:
    - image: np.ndarray, RGB iamge（H, W, 3）
    - temp: int, temperature（defult: 6500）
    - resize_max: int, longest image edge (default): 256）

    Returns:
    - wl: np.ndarray, wavelength (nm)
    - avg_spd: np.ndarray, average SPD intensity per pixel（Counts）
    """

    def resize_preserve_aspect(image, max_size=256):
        H, W, _ = image.shape
        scale = max_size / max(H, W)
        new_W = int(W * scale)
        new_H = int(H * scale)
        resized = cv2.resize(image, (new_W, new_H), interpolation=cv2.INTER_AREA)
        return resized
    
    def load_spd_from_npz(temp):
        cache_root = "./count_blue_light_energy/rgb2spd_lookup"
        mode = 'hauwei' 
        subdir = 'float64'
        filename = f"{mode}_{temp}K.npz"
        cache_path = os.path.join(cache_root, subdir, filename)
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"[ERROR] SPD cache not found: {cache_path}")
        data = np.load(cache_path)
        return data['wl'], data['spd_R'], data['spd_G'], data['spd_B']

    def RGB_to_SPD(rgb, spd_R, spd_G, spd_B):
        gamma = 2.24 
        r, g, b = rgb

        pred = (
            np.power((spd_R * (r / 255)), gamma) +
            np.power((spd_G * (g / 255)), gamma) +
            np.power((spd_B * (b / 255)), gamma)
        )
        return pred

    # Step 0: Load image
    image = cv2.imread(image_path)                  # BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 轉為 RGB
    
    # Step 1: Resize image
    image_resize = resize_preserve_aspect(image, max_size=resize_max)

    # Step 2: Load SPD mapping
    wl, spd_R, spd_G, spd_B = load_spd_from_npz(temp)

    # Step 3: Compute average SPD
    H, W, _ = image_resize.shape
    total_spd = np.zeros_like(spd_R, dtype=np.float64)
    total_blue_energy = 0.0
    blue_mask = (wl >= 450) & (wl <= 525)

    for y in range(H):
        for x in range(W):
            rgb = image_resize[y, x]
            spd = RGB_to_SPD(rgb, spd_R, spd_G, spd_B)
            total_spd += spd
            blue_energy = np.trapezoid(spd[blue_mask], wl[blue_mask])
            total_blue_energy += blue_energy
    # avg_blue_energy = total_blue_energy / (H * W)
    avg_spd = total_spd / (H * W) # per-pixel average SPD

    return wl, avg_spd
