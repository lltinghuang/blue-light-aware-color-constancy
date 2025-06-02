import argparse
import os
import time

import cv2
import numpy as np


# === CLI Argument Parser ===
def parse_arguments():
    parser = argparse.ArgumentParser(description="Estimate total blue light exposure from an image.")
    parser.add_argument('--image', type=str, required=True, help='Path to input image (e.g., ./image.png)')
    
    # Optional arguments
    parser.add_argument('--temp', type=int, default=6500, help='Screen temperature in K (e.g., 2700)')
    parser.add_argument('--mode', type=str, default='hauwei', help='Display mode (e.g., hauwei)')
    parser.add_argument('--precision', type=str, choices=['float32', 'float64'], default='float64', help='Precision of SPD data')
    parser.add_argument('--resize', type=int, default=256, help='Resize longest image edge to this size (default: 512), Use 0 to disable resizing.')
    parser.add_argument('--save_spd', nargs='?', const=True, default=None, help='Save average SPD as CSV. Use no value for auto-naming, or provide path.')
    parser.add_argument('--plot_spd', nargs='?', const=True, default=None, help='Save SPD plot as PNG. Use no value for auto-naming, or provide path.')

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
    gamma = 2.24  # Gamma correction value
    r, g, b = rgb

    pred = (
        np.power((spd_R * (r / 255)), gamma) +
        np.power((spd_G * (g / 255)), gamma) +
        np.power((spd_B * (b / 255)), gamma)
    )
    return pred


# === Estimate total blue light energy ===
def estimate_blue_light_energy(image, wl, spd_R, spd_G, spd_B, blue_band=(450, 525)):
    blue_mask = (wl >= blue_band[0]) & (wl <= blue_band[1])
    total_blue_energy = 0.0
    total_spd = np.zeros_like(spd_R, dtype=np.float64)

    H, W, _ = image.shape
    for y in range(H):
        for x in range(W):
            rgb = image[y, x]
            spd = RGB_to_SPD(rgb, spd_R, spd_G, spd_B)
            total_spd += spd
            blue_energy = np.trapezoid(spd[blue_mask], wl[blue_mask])
            total_blue_energy += blue_energy
    avg_blue_energy = total_blue_energy / (H * W)
    avg_spd = total_spd / (H * W)
    # avg_spd = total_spd
    return total_blue_energy, avg_blue_energy, avg_spd

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

# 0531 update: save spd results to a CSV file
def save_spd_to_csv(wl, spd, output_path):
    import pandas as pd
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame({'wavelength': wl, 'intensity': spd})
    df.to_csv(output_path, index=False)
    print(f"[INFO] Saved average SPD to: {output_path}")

def plot_spd_curve(wl, spd, output_path):
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(wl, spd, label="Average SPD", color="red")
    plt.title("Spectral Power Distribution")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[INFO] SPD plot saved to: {output_path}")


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
    total_energy, avg_energy, avg_spd = estimate_blue_light_energy(image, wl, spd_R, spd_G, spd_B)
    elapsed = time.time() - start_time
    # print("[DEBUG] First pixel RGB:", image[0, 0])

    print(f"[RESULT] Total blue light energy (450–525nm): {total_energy:.4f} (counts⋅nm)")
    print(f"[RESULT] Average blue energy per-pixel      : {avg_energy:.6f} (counts⋅nm)")
    print(f"[INFO] Execution time: {elapsed:.3f} seconds")

    # 0531 update: save spd results to a CSV file
    # === Handle SPD saving ===
    if args.save_spd:
        if args.save_spd is True:
            # 自動命名
            base = os.path.splitext(os.path.basename(args.image))[0]
            output_path = os.path.join("test_spd_result", f"{base}_spd.csv")
        else:
            # 指定路徑
            output_path = args.save_spd
        save_spd_to_csv(wl, avg_spd, output_path)

    # === Handle SPD plotting ===
    if args.plot_spd:
        if args.plot_spd is True:
            base = os.path.splitext(os.path.basename(args.image))[0]
            output_path = os.path.join("test_spd_result", f"{base}_spd_plot.png")
        else:
            output_path = args.plot_spd
        plot_spd_curve(wl, avg_spd, output_path)

if __name__ == '__main__':
    main()
