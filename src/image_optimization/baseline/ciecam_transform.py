import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
from colour import (CCS_ILLUMINANTS, CIECAM02_to_XYZ, RGB_to_XYZ,
                    XYZ_to_CIECAM02, XYZ_to_sRGB, sRGB_to_XYZ, xy_to_xyY,
                    xyY_to_XYZ)
from colour.colorimetry import SpectralShape, sd_blackbody
from colour.colorimetry.tristimulus_values import sd_to_XYZ_integration
from colour.temperature import CCT_to_xy_CIE_D
from PIL import Image

from img_transform_temp import convert_K_to_RGB, linear_to_srgb, srgb_to_linear

SPECTRAL_SHAPE_RAWTOACES: SpectralShape = SpectralShape(380, 780, 5)


def apply_inverse_color_temperature(image: np.ndarray, target_temp: float) -> Image.Image:
    # [TODO] Fix this function 
    scaling = convert_K_to_RGB(target_temp)  # sRGB [0,1]
    scaling_lin = srgb_to_linear(scaling)
    # avoid deviding by 0
    scaling_lin = np.clip(scaling_lin, 1e-6, None)
    print(f"linear scale {scaling_lin}")
    img_np = np.asarray(image).astype(np.float32)
    img_lin = srgb_to_linear(img_np)
    
    # Undo the scaling in linear space
    img_lin[..., 0] /= scaling_lin[0]
    img_lin[..., 1] /= scaling_lin[1]
    img_lin[..., 2] /= scaling_lin[2]
    # Clamp and convert back to sRGB
    img_lin = np.clip(img_lin, 0.0, 1.0)
    img_srgb = linear_to_srgb(img_lin)
    
    img_srgb = np.clip(img_srgb, 0.0, 1.0)
    
    return img_srgb

def get_XYZ_white_from_temperature(temp: float) -> np.ndarray:
    """
    Estimate XYZ whitepoint from color temperature (CCT), using:
    - CIE D-series illuminants for CCT in [4000, 25000]
    - Blackbody radiator for CCT in [1000, 4000)
    """
    if 4000 <= temp <= 25000:
        # Use CIE D-series daylight model
        xy = CCT_to_xy_CIE_D(temp)
        XYZ = xyY_to_XYZ([*xy, 1.0])
        print(f"Daylight CCT {temp}K to xy: {xy} to XYZ: {XYZ}")
        return XYZ

    elif 1000 <= temp < 4000:
        # Use blackbody spectral distribution → integrate to XYZ
        sd = sd_blackbody(temp, SPECTRAL_SHAPE_RAWTOACES)
        XYZ = sd_to_XYZ_integration(sd)
        XYZ /= XYZ[1]  # Normalize Y to 1
        print(f"Blackbody CCT {temp}K to XYZ: {XYZ}")
        return XYZ

    else:
        raise ValueError("CCT must be between 1000K and 25000K for reliable results.")

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_image(rgb_image, path):
    rgb_uint8 = np.clip(rgb_image * 255, 0, 255).astype(np.uint8)
    Image.fromarray(rgb_uint8).save(path)

def simulate_CIECAM02(rgb_image, Y_n=20, temp=2700, surround='Average'):
    rgb_image = rgb_image / 255.0
    
    original_shape = rgb_image.shape

    # Step 1: Flatten and extract unique colors
    rgb_flat = rgb_image.reshape(-1, 3)
    unique_colors, inverse_indices = np.unique(rgb_flat, axis=0, return_inverse=True)

    # Step 2: Convert to XYZ
    xyz_unique = np.array([sRGB_to_XYZ(rgb) * 100 for rgb in unique_colors])

    # White points
    XYZ_w_d65 = xyY_to_XYZ(xy_to_xyY(CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]))
    print(f"The given xyz = {XYZ_w_d65}")
    get_XYZ_white_from_temperature(6500)
    # LED-B1	0.4560	0.4560 (wiki)
    XYZ_w_dxx = get_XYZ_white_from_temperature(temp)

    from colour.appearance import VIEWING_CONDITIONS_CIECAM02
    vc = VIEWING_CONDITIONS_CIECAM02[surround]
    Y_b = 20.0        # background luminance
    # Step 4: Adapt each unique color
    cam02_d65 = [XYZ_to_CIECAM02(x, XYZ_w_d65, Y_n, Y_b, surround=vc) for x in xyz_unique]
    xyz_dxx = [CIECAM02_to_XYZ(a, XYZ_w_dxx, Y_n, Y_b, surround=vc) for a in cam02_d65]
    rgb_dxx = np.stack([XYZ_to_sRGB(x / 100) for x in xyz_dxx])

    # Step 5: Map back to full image
    rgb_output = rgb_dxx[inverse_indices].reshape(original_shape)
    
    return np.clip(rgb_output, 0, 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', required=True, help='Input image path')
    parser.add_argument('--out', default='Temperature_expected.png', help='Output path for D27 image')
    parser.add_argument('--out_optimized', default='Program_optimized.png', help='Output path for D27 image')
    parser.add_argument('--temp', type=float, default=2700, help='Input display color temperature (e.g., 2700 for warm mode)')

    args = parser.parse_args()

    # Load image
    print(f"Loading image: {args.img}")
    img = load_image(args.img)

    # Simulate D27 appearance
    print("Simulating perceptual appearance under D27...")
    img_expected = simulate_CIECAM02(img, temp=args.temp)
    save_image(img_expected, args.out)
    img_opt = apply_inverse_color_temperature(img_expected, args.temp)
    save_image(img_opt, args.out_optimized)
    
    print(f"Saved simulated image to {args.out}")
    print(f"Saved Optimized image to {args.out_optimized}")


if __name__ == '__main__':
    main()
