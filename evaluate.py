import argparse
import cv2
import numpy as np
import sys
import pandas as pd
from scipy.spatial import ConvexHull
from skimage.metrics import structural_similarity as compare_ssim
import colour  # install colour-science
from colour.difference import delta_E_CIE2000
from colour.colorimetry import SDS_LEFS_PHOTOPIC
from colour import SpectralShape
from spd_utils import extract_avg_spd_from_image

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def delta_lightness(img1, img2):
    """Compute average lightness difference between two sRGB images."""
    def compute_L(image):
        XYZ = colour.sRGB_to_XYZ(image / 255.0)
        Y = XYZ[..., 1]
        y_ratio = Y / 1.0
        # Apply the CIE lightness formula
        L = np.where(y_ratio > 0.008856,
                     116 * np.cbrt(y_ratio) - 16,
                     903.3 * y_ratio)
        return np.mean(L)
    
    return abs(compute_L(img1) - compute_L(img2))

def XYZ_to_uv_prime(XYZ):
    """
    Convert CIE XYZ to CIE 1976 u'v' chromaticity coordinates.
    """
    X, Y, Z = np.moveaxis(XYZ, -1, 0)
    denominator = X + 15 * Y + 3 * Z + 1e-10  # Avoid divide by zero
    u_prime = (4 * X) / denominator
    v_prime = (9 * Y) / denominator
    return u_prime, v_prime

def compute_uv_prime(image):
    """
    Compute average CIE 1976 u'v' coordinates from an sRGB image.
    """
    XYZ = colour.sRGB_to_XYZ(image.astype(np.float32) / 255.0)
    u_prime, v_prime = XYZ_to_uv_prime(XYZ)
    return np.mean(u_prime), np.mean(v_prime)

def compute_delta_uv_prime(image1, image2):
    """
    Compute Δu'v' between two images.
    """
    u1, v1 = compute_uv_prime(image1)
    u2, v2 = compute_uv_prime(image2)
    return np.sqrt((u2 - u1) ** 2 + (v2 - v1) ** 2)

def compute_duv(image):
    u_prime, v_prime = compute_uv_prime(image)
    u = u_prime
    v = 2 * v_prime / 3
    _, Duv = colour.uv_to_CCT((u, v), method='Ohno 2013', return_D_uv=True)
    return Duv

def compute_eml(image_path, temp=6500):
    # Load SPD
    wl_spd, intensity_spd = extract_avg_spd_from_image(image_path, temp, resize_max=256)

    # spectral irradiance => (W/m^2/nm), we should enable "intensity correction" QQ
    lef = SDS_LEFS_PHOTOPIC['CIE 1924 Photopic Standard Observer']
    lef = lef.copy().align(SpectralShape(start=wl_spd.min(), end=wl_spd.max(), interval=1))
    lef_interp = np.interp(wl_spd, lef.wavelengths, lef.values)
    # Simulated luminance from counts:
    target_luminance = 300
    L_counts = 683 * np.trapezoid(intensity_spd * lef_interp, wl_spd)
    scale = target_luminance / L_counts
    irradiance_spd = intensity_spd * scale  # now in W/m²/nm

    # Load melanopic sensitivity
    melanopic_csv_path = 'melanopic.csv'
    mel_df = pd.read_csv(melanopic_csv_path)
    wl_mel = mel_df['nm'].values
    mel_sens = mel_df['melanopic'].values

    # Interpolate SPD to melanopic wavelengths
    interp_spd = np.interp(wl_mel, wl_spd, irradiance_spd)

    # Integrate using trapezoidal rule
    eml = 72983.25 * np.trapezoid(interp_spd * mel_sens, wl_mel)

    # print("SPD max:", np.max(irradiance_spd))
    # print("SPD mean:", np.mean(irradiance_spd))
    # print("Melanopic sensitivity max:", np.max(mel_sens))
    # print("EML value (unitless):", eml)

    return eml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_image', help='Path to the reference image')
    parser.add_argument('--image', help='Path to the transformed image')
    parser.add_argument('--temp', type=int, default=6500, help='Color temperature for EML calculation (default: 6500K)')
    parser.add_argument('--metric', required=True, choices=['delta_lightness', 'CIEDE2000', 'delta_uv_prime', 'Duv', 'EML'], help='Metric to use for evaluation')
    args = parser.parse_args()

    metrics_require_ref = ['delta_lightness', 'CIEDE2000', 'delta_uv_prime']
    if args.metric in metrics_require_ref and not args.ref_image:
        print(f"Error: --ref_image is required for metric {args.metric}")
        sys.exit(1)

    if args.metric in metrics_require_ref:
        ref = load_image(args.ref_image)
    img = load_image(args.image)

    if args.metric == 'delta_lightness':
        score = delta_lightness(ref, img)

    elif args.metric == 'CIE2000':
        score = np.mean(delta_E_CIE2000(ref, img))

    elif args.metric == 'delta_uv_prime':
        score = compute_delta_uv_prime(ref, img)

    elif args.metric == 'Duv':
        score = compute_duv(img)

    elif args.metric == 'EML':
        temp = args.temp if hasattr(args, 'temp') else 6500
        score = compute_eml(args.image, temp)

    else:
        raise ValueError(f"Unsupported metric: {args.metric}")

    print(f"{args.metric} score: {score:.4f}")


if __name__ == '__main__':
    main()
