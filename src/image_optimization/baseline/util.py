import cv2
import numpy as np
from PIL import Image

from img_transform_temp import (apply_color_temperature, convert_K_to_RGB,
                                linear_to_srgb, srgb_to_linear)

# This follows the suggestions in the paper:
# "Exploiting Perceptual Anchoring for Color Image Enhancement"

# Device Characteristics =================================
# High light environment
gamma_rf, gamma_gf, gamma_bf = 2.4767, 2.4286, 2.3792
M_f = np.array([
    [95.57,  64.67,  33.01],
    [49.49, 137.29,  14.76],
    [ 0.44,  27.21, 169.83]
])

# Low light environment
gamma_rl, gamma_gl, gamma_bl = 2.2212, 2.1044, 2.1835
M_l = np.array([
    [4.61, 3.35, 1.78],
    [2.48, 7.16, 0.79],
    [0.28, 1.93, 8.93]
])
# =========================================================

def RGBs_to_XYZ(images: np.ndarray, light_mode: bool = True) -> np.ndarray:
    """
    Vectorized version of RGB_to_XYZ for multiple RGB values.

    Args:
        images (np.ndarray): N x 3 array of RGB values in [0, 1].
        light_mode (bool): True for high light, False for low light.

    Returns:
        np.ndarray: N x 3 array of XYZ values.
    """
    gamma_r, gamma_g, gamma_b = (gamma_rf, gamma_gf, gamma_bf) if light_mode else (gamma_rl, gamma_gl, gamma_bl)
    M = M_f if light_mode else M_l

    # Apply per-channel gamma correction
    images_lin = np.zeros_like(images)
    images_lin[:, 0] = images[:, 0] ** gamma_r
    images_lin[:, 1] = images[:, 1] ** gamma_g
    images_lin[:, 2] = images[:, 2] ** gamma_b

    # Matrix transform (dot each RGB with M.T)
    xyz = np.dot(images_lin, M.T)
    return xyz

def XYZ_to_RGB(image: np.ndarray, light_mode: bool = True)-> np.ndarray:
    """
    Converts a device-specific XYZ image back to RGB using inverse matrix and inverse gamma correction.

    Args:
        image (np.ndarray): Input image (H x W x 3) in XYZ space.
        light_mode (bool): True for high light condition, False for low light.

    Returns:
        np.ndarray: Reconstructed RGB image (H x W x 3), clipped to [0, 1].
    """
    gamma_r, gamma_g, gamma_b = (gamma_rf, gamma_gf, gamma_bf) if light_mode else (gamma_rl, gamma_gl, gamma_bl)
    M = M_f if light_mode else M_l
    M_inv = np.linalg.inv(M)

    # Apply inverse transformation
    rgb_lin = np.dot(M_inv, image.reshape(-1, 3).T).T.reshape(image.shape)

    # Clip negative values from numerical artifacts
    rgb_lin = np.clip(rgb_lin, 0, None)

    # Apply inverse gamma
    rgb = np.zeros_like(rgb_lin)
    rgb[..., 0] = rgb_lin[..., 0] ** (1 / gamma_r)
    rgb[..., 1] = rgb_lin[..., 1] ** (1 / gamma_g)
    rgb[..., 2] = rgb_lin[..., 2] ** (1 / gamma_b)

    # Clip final output to [0, 1] and warn if needed
    if np.any((rgb < 0) | (rgb > 1)):
        print("Warning: XYZ to RGB conversion produced out-of-range values. Clipping applied.")
        rgb = np.clip(rgb, 0, 1)

    return rgb


def apply_color_temperature_np(image_np: np.ndarray, target_temp: float) -> np.ndarray:
    """
    Apply a physically-based white balance shift to an image in NumPy array format.

    Args:
        image_np (np.ndarray): Input image as a NumPy array, shape (H, W, 3), values in [0, 1].
        target_temp (float): Target color temperature in Kelvin.

    Returns:
        np.ndarray: White-balance-adjusted image, same shape and dtype, values in [0, 1].
    """
    # Get energy scaling at target and reference color temperature
    target_rgb = convert_K_to_RGB(target_temp)
    scaling = srgb_to_linear(target_rgb)
    print(f"new scaling: {scaling}")
    
    # Ensure input is float32 in [0, 1]
    image_np = image_np.astype(np.float32)
    image_np = np.clip(image_np, 0.0, 1.0)

    # Convert to linear RGB
    img_lin = srgb_to_linear(image_np)

    # Apply energy scaling in linear space
    img_lin[..., 0] *= scaling[0]
    img_lin[..., 1] *= scaling[1]
    img_lin[..., 2] *= scaling[2]

    # Convert back to sRGB and clip to valid range
    img_srgb = linear_to_srgb(np.clip(img_lin, 0.0, 1.0))
    img_srgb = np.clip(img_srgb, 0.0, 1.0)

    return img_srgb


def apply_inverse_color_temperature(image: np.ndarray, target_temp: float) -> Image.Image:
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
    img_srgb = linear_to_srgb(img_lin)
    
    img_srgb = np.clip(img_srgb, 0.0, 1.0)
    
    return img_srgb

def xyz_to_rgb(xyz):
    # Linear transformation
    xyz = np.array(xyz)  # Convert to NumPy array for easier calculations
    rgb = np.array([3.2404542 * xyz[0] - 1.5371385 * xyz[1] - 0.4985314 * xyz[2],
                    -0.9692660 * xyz[0] + 1.8760108 * xyz[1] + 0.0415560 * xyz[2],
                    0.0556434 * xyz[0] - 0.2040259 * xyz[1] + 1.0572252 * xyz[2]])
    # Gamma correction (sRGB)
    for i in range(3):
        if rgb[i] <= 0.0031308:
            rgb[i] = 12.92 * rgb[i]
        else:
            rgb[i] = 1.055 * (rgb[i] ** (1 / 2.4)) - 0.055
    return np.clip(rgb, 0, 1)

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_image(rgb_image, path):
    rgb_uint8 = np.clip(rgb_image * 255, 0, 255).astype(np.uint8)
    Image.fromarray(rgb_uint8).save(path)

def correctness_check():
    image = Image.open("../../img_transform_temp/img/cute_furret_small.png").convert("RGB")
    transformed_image = apply_color_temperature(image, target_temp = 3500)
    transformed_image = np.asarray(transformed_image).astype(np.float32) / 255
    transformed_image = apply_inverse_color_temperature(transformed_image, target_temp = 3500)
    save_image(transformed_image, "Hello.png")

def violation_check(rgb_image: np.ndarray, temp: float):
    limit = convert_K_to_RGB(temp)  # e.g., [R_max, G_max, B_max] in [0,1] range
    if np.max(rgb_image) > 2:
        rgb_image = rgb_image / 255.0  # normalize if image is in [0,255]
    
    # Check for violations
    violations = rgb_image > limit  # shape: (H, W, 3) > (3,)
    violation_mask = np.any(violations, axis=-1)  # shape: (H, W), True if any channel exceeds

    # Count total and per-channel violations
    total_violations = np.sum(violation_mask)
    channel_violations = np.sum(violations, axis=(0, 1))

    print(f"Total violating pixels: {total_violations}")
    print(f"Per-channel violations: R={channel_violations[0]}, G={channel_violations[1]}, B={channel_violations[2]}")

    return violation_mask  # Optional: return mask to visualize or process further
    