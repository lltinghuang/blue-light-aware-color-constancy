import argparse
import os
import time

import cv2
import numpy as np


# === CLI Argument Parser ===
def parse_arguments():
    parser = argparse.ArgumentParser(description="Estimate total blue light exposure from an image.")
    parser.add_argument('--image', type=str, required=True, help='Path to input image (e.g., ./image.png)')
    parser.add_argument('--temp', type=int, required=True, help='Screen temperature in K (e.g., 2700)')
    
    # Optional arguments
    parser.add_argument('--mode', type=str, default='hauwei', help='Display mode (e.g., hauwei, sRGB)')
    parser.add_argument('--precision', type=str, choices=['float32', 'float64'], default='float32', help='Precision of SPD data')
    parser.add_argument('--resize', type=int, default=256, help='Resize longest image edge to this size (default: 512), Use 0 to disable resizing.')
    return parser.parse_args()

# === Load SPD data from npz file ===
def load_spd_from_npz(temp, mode, precision='float32'):
    cache_root = "./count_blue_light_energy/rgb2spd_lookup"
    subdir = 'float32' if precision == 'float32' else 'float64'
    filename = f"{mode}_{temp}K.npz"
    cache_path = os.path.join(cache_root, subdir, filename)
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"[ERROR] SPD cache not found: {cache_path}")
    data = np.load(cache_path)
    return data['wl'], data['spd_R'], data['spd_G'], data['spd_B']

# === RGB → SPD ===
def RGB_to_SPD(rgb, spd_R, spd_G, spd_B):
    R, G, B = rgb
    return (R / 255.0) * spd_R + (G / 255.0) * spd_G + (B / 255.0) * spd_B

# === Estimate total blue light energy ===
def estimate_blue_light_energy(image, wl, spd_R, spd_G, spd_B, blue_band=(450, 525)):
    blue_mask = (wl >= blue_band[0]) & (wl <= blue_band[1])
    total_blue_energy = 0.0

    H, W, _ = image.shape
    for y in range(H):
        for x in range(W):
            rgb = image[y, x]
            spd = RGB_to_SPD(rgb, spd_R, spd_G, spd_B)
            blue_energy = np.trapezoid(spd[blue_mask], wl[blue_mask])
            total_blue_energy += blue_energy
    avg_blue_energy = total_blue_energy / (H * W)
    return total_blue_energy, avg_blue_energy

# === Loads an image and converts to RGB ===
def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Failed to load image.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# === Resizes the image size (for faster processing) ===
def resize_preserve_aspect(image, max_size=512):
    H, W, _ = image.shape
    scale = max_size / max(H, W)
    new_W = int(W * scale)
    new_H = int(H * scale)
    resized = cv2.resize(image, (new_W, new_H), interpolation=cv2.INTER_AREA)
    return resized

# === main ===
def main():
    args = parse_arguments()

    # print(f"[INFO] Loading image: {args.image}")
    image = load_image(args.image)
    H, W, _ = image.shape
    num_pixels = H * W
    print(f"[INFO] Image size: {H}x{W} = {num_pixels} pixels")

    if args.resize > 0:
        image = resize_preserve_aspect(image, max_size=args.resize)
        H, W, _ = image.shape
        print(f"[INFO] Resized image to: {H}x{W} = {H * W} pixels")
    else:
        print("[INFO] Resizing disabled")

    # print(f"[INFO] Loading SPD: mode={args.mode}, temp={args.temp}K, precision={args.precision}")
    wl, spd_R, spd_G, spd_B = load_spd_from_npz(args.temp, args.mode, args.precision)

    start_time = time.time()
    # print(f"[INFO] Estimating blue light energy...")
    total_energy, avg_energy = estimate_blue_light_energy(image, wl, spd_R, spd_G, spd_B)
    elapsed = time.time() - start_time

    print(f"[RESULT] Total blue light energy (450–525nm): {total_energy:.4f} (counts⋅nm)")
    print(f"[RESULT] Average per-pixel blue energy     : {avg_energy:.6f} (counts⋅nm)")
    print(f"[INFO] Execution time: {elapsed:.3f} seconds")

if __name__ == '__main__':
    main()
