import argparse

import matplotlib.pyplot as plt
import numpy as np
from colour import (CIECAM02_to_XYZ, XYZ_to_CIECAM02, XYZ_to_sRGB, sRGB_to_XYZ,
                    xyY_to_XYZ)
from colour.colorimetry import SpectralShape, sd_blackbody
from colour.colorimetry.tristimulus_values import sd_to_XYZ_integration
from colour.plotting import plot_chromaticity_diagram_CIE1931
from colour.temperature import CCT_to_xy_CIE_D
from PIL import Image
from util import (apply_inverse_color_temperature, color_scaler, load_image,
                  normalize_scaler, save_image, violation_check, xyz_to_rgb)

from img_transform_temp import (apply_color_temperature, convert_K_to_RGB,
                                linear_to_srgb, srgb_to_linear)


def get_XYZ_white_from_temperature(temp: float) -> np.ndarray:
    """
    Estimate XYZ whitepoint from color temperature (CCT), using:
    - CIE D-series illuminants for CCT in [4000, 25000]
    - Blackbody radiator for CCT in [1000, 4000)
    """
    if 4000 <= temp <= 25000:
        # Use CIE D-series daylight model
        xy = CCT_to_xy_CIE_D(temp)
        XYZ = xyY_to_XYZ([*xy, 1.0])  # Y normalized to 1.0
    elif 1000 <= temp < 4000:
        # Use blackbody radiator model
        spectral_shape = SpectralShape(360, 830, 1)
        sd = sd_blackbody(temp, spectral_shape)
        XYZ = sd_to_XYZ_integration(sd)
        XYZ /= XYZ[1]  # Normalize Y to 1
    else:
        raise ValueError("CCT must be in the range [1000, 25000] K")

    print(f"Temperature: {temp}K → XYZ: {XYZ}")
    return XYZ


# Brute force algor
def simulate_white_under_CCT(temp: float) -> np.ndarray:
    """
    Returns the best-fit [X, 1.0, Z] whitepoint that maps to the given RGB white under a different CCT.
    """
    # Get the target sRGB white under this CCT (gamma-encoded)
    objective = convert_K_to_RGB(temp)

    best_candidate = None
    best_energy = float('inf')

    for i in np.arange(0.0, 2.0, 0.005):
        for j in np.arange(0.0, 2.0, 0.005):
            candidate = np.array([i, 1.0, j])  # fixed Y=1
            result = xyz_to_rgb(candidate)
            current_energy = np.linalg.norm(result - objective)
            if current_energy < best_energy:
                best_candidate = candidate
                best_energy = current_energy

    return best_candidate

def simulate_CIECAM02(rgb_image, Y_n=20, temp=2700, surround='Average'):
    rgb_image = rgb_image / 255.0
    
    original_shape = rgb_image.shape

    # Step 1: Flatten and extract unique colors
    rgb_flat = rgb_image.reshape(-1, 3)
    unique_colors, inverse_indices = np.unique(rgb_flat, axis=0, return_inverse=True)

    # Step 2: Convert to XYZ
    xyz_unique = np.array([sRGB_to_XYZ(rgb) * 100 for rgb in unique_colors])

    # White points
    '''
    # This approach would generate a violation picture that the color exceed the printable area
    it would required 
    1. Gumat mapping or tone mapping to fit in the color space
    XYZ_w_d65 = get_XYZ_white_from_temperature(6500)
    XYZ_w_dxx = get_XYZ_white_from_temperature(temp)
    '''
    
    # This approach on the other hand, lead to a valid space, however, it's generally more brown ?? 
    XYZ_w_d65 = get_XYZ_white_from_temperature(6500)
    XYZ_w_dxx = get_XYZ_white_from_temperature(temp)

    from colour.appearance import VIEWING_CONDITIONS_CIECAM02
    vc = VIEWING_CONDITIONS_CIECAM02[surround]
    Y_b = 20.0        # background luminance
    # Step 4: Adapt each unique color
    cam02_d65 = [XYZ_to_CIECAM02(x, XYZ_w_d65, Y_n, Y_b, surround=vc) for x in xyz_unique]
    xyz_dxx = [CIECAM02_to_XYZ(a, XYZ_w_dxx, Y_n, Y_b, surround=vc) for a in cam02_d65]
    
    # xyz_dxx = np.clip(xyz_dxx, 0 , 100)
    rgb_dxx = np.stack([XYZ_to_sRGB(x / 100) for x in xyz_dxx])

    # Step 5: Map back to full image
    rgb_output = rgb_dxx[inverse_indices].reshape(original_shape)
    
    # calculate the write point shift
    true_white = XYZ_to_CIECAM02(np.array([1.0,1.0,1.0]), XYZ_w_d65, Y_n, Y_b, surround=vc)
    resulting_white = CIECAM02_to_XYZ(true_white, XYZ_w_dxx, Y_n, Y_b, surround=vc)
    
    return resulting_white , np.clip(rgb_output, 0, 1)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', required=True, help='Input image path')
    parser.add_argument('--out', default='Temperature_expected.png', help='Output path for D27 image')
    parser.add_argument('--out_optimized', default='Program_optimized.png', help='Output path for D27 image')
    parser.add_argument('--temp', type=float, default=4500, help='Input display color temperature (e.g., 2700 for warm mode)')

    args = parser.parse_args()

    # Load image
    img = load_image(args.img)
    
    print(f"Simulating perceptual appearance under temperature of {args.temp}...")
    w_point , img_expected = simulate_CIECAM02(img, temp=args.temp)
    
    
    '''
    It might need somehow a more cleaver way to handle
    '''
    # print(f"in the resulting image, the max rgb = {np.max(img_expected[:,:,0])}, {np.max(img_expected[:,:,1])}, {np.max(img_expected[:,:,2])}")
    '''
    
    # This part is part of experimental setup aim to make it fit in the scale beforehead.
    
    max_val = srgb_to_linear(np.array([np.max(img_expected[:,:,0]), np.max(img_expected[:,:,1]),np.max(img_expected[:,:,2])]))
    display_limit_lin = srgb_to_linear(convert_K_to_RGB(args.temp))
    # elementwise div
    correction = normalize_scaler(max_val / display_limit_lin)
    
    print(f"corerection : {correction}")
    img_expected = color_scaler(img_expected, correction)'''
    # save here or after the correction
    save_image(img_expected, args.out)
    
    '''
    voilation check on the invalid points, report number of points that can't be shown in that temperature setup.
    '''
    violation_check(rgb_image = img_expected, temp = args.temp)
    img_opt = apply_inverse_color_temperature(img_expected, args.temp)
    save_image(img_opt, args.out_optimized)
    
    print(f"Saved simulated image to {args.out}")
    print(f"Saved Optimized image to {args.out_optimized}")


if __name__ == '__main__':
    main()
