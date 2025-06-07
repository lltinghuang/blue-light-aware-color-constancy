import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
from colour import (CIECAM02_to_XYZ, CIECAM16_to_XYZ, XYZ_to_CIECAM02,
                    XYZ_to_CIECAM16, xyY_to_XYZ)
from colour.colorimetry import SpectralShape, sd_blackbody
from colour.colorimetry.tristimulus_values import sd_to_XYZ_integration
from colour.plotting import plot_chromaticity_diagram_CIE1931
from colour.temperature import CCT_to_xy_CIE_D
from guided_filter.core.filter import GuidedFilter
from joblib import Parallel, delayed
from PIL import Image
from util import (RGBs_to_XYZ, XYZ_to_RGB, apply_color_temperature_np,
                  load_image, save_image, violation_check)

from img_transform_temp import apply_color_temperature, convert_K_to_RGB


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

def simulate_CIECAM02(rgb_image, Y_n=20, temp=2700, surround='Average'):
    '''
    Input range (0 255)
    Output range (0 1)
    '''
    rgb_image = rgb_image / 255.0
    
    original_shape = rgb_image.shape

    # Step 1: Flatten and extract unique colors
    rgb_flat = rgb_image.reshape(-1, 3)
    unique_colors, inverse_indices = np.unique(rgb_flat, axis=0, return_inverse=True)

    # Step 2: Convert to XYZ
    xyz_unique = RGBs_to_XYZ(unique_colors, light_mode=True)
    
    # White points
    '''
    # This approach would generate a violation picture that the color exceed the printable area
    it would required 
    1. Gumat mapping or tone mapping to fit in the color space
    XYZ_w_d65 = get_XYZ_white_from_temperature(6500)
    XYZ_w_dxx = get_XYZ_white_from_temperature(temp)
    '''
    
    # This approach on the other hand, lead to a valid space, however, it's generally more brown ?? 
    XYZ_w_d65 = RGBs_to_XYZ(convert_K_to_RGB(6500).reshape(1,3)).flatten()
    XYZ_w_dxx = RGBs_to_XYZ(convert_K_to_RGB(temp).reshape(1,3)).flatten()
    from colour.appearance import VIEWING_CONDITIONS_CIECAM02
    vc = VIEWING_CONDITIONS_CIECAM02[surround]
    Y_b = 20.0        # background luminance
    # Step 4: Adapt each unique color
    # print(f"first xyz unique {xyz_unique[0]}")
    # print(f"reverse {XYZ_to_CIECAM02(xyz_unique[0], XYZ_w_d65, Y_n, Y_b, surround=vc)}")
    print(f"start transform with number {len(xyz_unique)}")
    # Step 4: Adapt each unique color using joblib for parallel speedup
    cam02_d65 = Parallel(n_jobs=6, backend="threading")(
        delayed(XYZ_to_CIECAM16)(x, XYZ_w_d65, Y_n, Y_b, surround=vc) for x in xyz_unique
    )

    xyz_dxx = Parallel(n_jobs=6, backend="threading")(
        delayed(CIECAM16_to_XYZ)(a, XYZ_w_dxx, Y_n, Y_b, surround=vc) for a in cam02_d65
    )
    
    '''
    # cam 02
    cam02_d65 = Parallel(n_jobs=6, backend="threading")(
        delayed(XYZ_to_CIECAM02)(x, XYZ_w_d65, Y_n, Y_b, surround=vc) for x in xyz_unique
    )

    xyz_dxx = Parallel(n_jobs=6, backend="threading")(
        delayed(CIECAM02_to_XYZ)(a, XYZ_w_dxx, Y_n, Y_b, surround=vc) for a in cam02_d65
    )
    '''
    
    print("end transform")

    
    # Compute JC per unique color
    JC_unique = np.array([cam.J * cam.C for cam in cam02_d65])  # shape: [num_unique,]

    # Reconstruct JC image per pixel
    JC_full = JC_unique[inverse_indices].reshape(original_shape[:-1])  # (H, W)

    # xyz_dxx = np.clip(xyz_dxx, 0 , 100)
    rgb_dxx = np.stack([XYZ_to_RGB(x) for x in xyz_dxx])

    # Step 5: Map back to full image
    rgb_output = rgb_dxx[inverse_indices].reshape(original_shape)

    return np.clip(rgb_output, 0, 1), JC_full


def apply_local_adaptation(image: np.ndarray, eta=36, lmda=10) -> np.ndarray:
    """
    Apply tone mapping and color gain scaling to enhance the input image.

    Args:
        image (np.ndarray): Input RGB image (uint8) with values in [0, 255].
        eta (float): Parameter controlling alpha scaling factor.
        lmda (float): Parameter controlling beta in tone mapping.

    Returns:
        np.ndarray: Enhanced RGB image (uint8) with values in [0, 255].
    """
    assert image.dtype == np.uint8, "Input image must be uint8"
    assert image.ndim == 3 and image.shape[2] == 3, "Input must be an RGB image"

    h, w = image.shape[:2]
    r, g, b = cv2.split(image)
    
    # Luminance
    l = 0.299 * r + 0.587 * g + 0.114 * b
    l = l.astype(np.float32) / 255.0  # Normalize to [0,1]
    
    # Local adaptation via Guided Filter
    GF = GuidedFilter(l, radius=30, eps=0.01)
    Hg = GF.filter(l)

    alpha = 1 + eta * l / np.max(l)
    Lgaver = np.exp(np.sum(np.log(0.001 + l)) / (h * w))
    beta = lmda * Lgaver
    Lout = alpha * np.log(l / Hg + beta)

    # Normalize Lout to [0, 255]
    Lout = cv2.normalize(Lout, None, 0, 255, cv2.NORM_MINMAX)

    # Compute gain map
    eps = 1e-6
    gain = Lout / (l * 255 + eps)
    gain[gain <= 0] = 0

    # Apply gain to each channel and normalize
    r_out = np.clip(r * gain, 0, None)
    g_out = np.clip(g * gain, 0, None)
    b_out = np.clip(b * gain, 0, None)

    r_out = (r_out / (np.max(r_out) + eps)) * 255
    g_out = (g_out / (np.max(g_out) + eps)) * 255
    b_out = (b_out / (np.max(b_out) + eps)) * 255

    # Merge and clip to valid 8-bit range
    merged = cv2.merge([
        np.clip(r_out, 0, 255),
        np.clip(g_out, 0, 255),
        np.clip(b_out, 0, 255)
    ])

    return cv2.convertScaleAbs(merged)

def post_process(pending_compensate_rgb : np.ndarray, original_rgb: np.ndarray, temp: float, JC_full: np.ndarray, do_map=True):
    compensate_img = pending_compensate_rgb / convert_K_to_RGB(temp)
    print(f"max after compensate {np.max(compensate_img)}")
    # doing two compensation at once ? 
    compensate_img = compensate_img / convert_K_to_RGB(temp)
    compensate_img = np.clip(compensate_img,0 ,1)
    if do_map:
        print("Post Gamut Mapping enabled")
        JC = JC_full  # shape (H, W)
        print(JC.shape)
        normalized_JC = (JC - np.min(JC)) / (np.max(JC) - np.min(JC))
        normalized_JC = np.expand_dims(normalized_JC, axis=-1)  # Shape: (H, W, 1)
        RGB_prime = original_rgb * normalized_JC + compensate_img * (1 - normalized_JC)
        return RGB_prime

    return compensate_img
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', required=True, help='Input image path')
    parser.add_argument('--out', default='Temperature_expected.png', help='Output path for D27 image')
    parser.add_argument('--out_optimized', default='Program_optimized.png', help='Output path for D27 image')
    parser.add_argument('--temp', type=float, default=2700, help='Input display color temperature (e.g., 2700 for warm mode)')

    args = parser.parse_args()

    # Load image
    img = load_image(args.img)
    # img = apply_local_adaptation(img)
    # preprocess img 
    preprocess_img = img / 255
    save_image(preprocess_img, "Preprocess_img.png")
    
    print(f"Simulating perceptual appearance under temperature of {args.temp}...")
    img_expected , JC_full = simulate_CIECAM02(img, temp=args.temp)
    
    post_process_img = post_process(pending_compensate_rgb = img_expected, original_rgb = img / 255,temp = args.temp, JC_full = JC_full, do_map = True)
    # save here or after the correction
    save_image(post_process_img, args.out_optimized)
    
    '''
    voilation check on the invalid points, report number of points that can't be shown in that temperature setup.
    '''
    img_out = apply_color_temperature_np(post_process_img, args.temp)
    save_image(img_out, args.out)
    
    print(f"Saved simulated image to {args.out}")
    print(f"Saved Optimized image to {args.out_optimized}")


if __name__ == '__main__':
    main()
