import cv2
import numpy as np
from PIL import Image

from img_transform_temp import (apply_color_temperature, convert_K_to_RGB,
                                linear_to_srgb, srgb_to_linear)


def normalize_scaler(scaler: np.ndarray) -> np.ndarray:
    """
    Normalize a per-channel scaler such that:
    - If a value > 1, replace it with its reciprocal (1 / value)
    - If a value <= 1, replace it with 1

    Parameters:
        scaler (np.ndarray): A 1D array of length 3 representing scaling factors
                             for the R, G, and B channels.

    Returns:
        np.ndarray: Normalized scaler array of the same shape.
    """
    normalized = np.where(scaler > 1, 1.0 / scaler, 1.0)
    return normalized

def color_scaler(image: np.ndarray, scaler: np.ndarray) -> np.ndarray:
    """
    Apply per-channel scaling to an sRGB image in linear RGB space.

    Parameters:
        image (np.ndarray): Input image in sRGB color space with shape (H, W, 3),
                            values expected in [0, 1].
        scaler (np.ndarray): A 1D array of length 3 representing scaling factors
                             for the R, G, and B channels, respectively.

    Returns:
        np.ndarray: The scaled image in sRGB space with values clipped to [0, 1].
    """
    # Convert sRGB to linear RGB
    image_lin = srgb_to_linear(image)

    # Apply per-channel scaling in linear space
    image_lin[..., 0] *= scaler[0]
    image_lin[..., 1] *= scaler[1]
    image_lin[..., 2] *= scaler[2]

    # Convert back to sRGB and clip to valid range
    img_srgb = linear_to_srgb(image_lin)
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
    