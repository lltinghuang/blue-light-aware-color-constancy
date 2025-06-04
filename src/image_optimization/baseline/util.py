import numpy as np
from PIL import Image

from img_transform_temp import convert_K_to_RGB, linear_to_srgb, srgb_to_linear


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
    img_lin = np.clip(img_lin, 0.0, 1.0)
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