import argparse
import os

import numpy as np
from PIL import Image


def convert_K_to_RGB(colour_temperature: float) -> np.ndarray:
    """
    Convert color temperature in Kelvin to an RGB value.
    Based on: http://www.tannerhelland.com/4435/convert-temperature-rgb-algorithm-code/
    """
    colour_temperature = np.clip(colour_temperature, 1000, 40000)
    tmp_internal = colour_temperature / 100.0

    if tmp_internal <= 66:
        red = 255
    else:
        red = 329.698727446 * (tmp_internal - 60)**-0.1332047592

    if tmp_internal <= 66:
        green = 99.4708025861 * np.log(tmp_internal) - 161.1195681661
    else:
        green = 288.1221695283 * (tmp_internal - 60)**-0.0755148492

    if tmp_internal >= 66:
        blue = 255
    elif tmp_internal <= 19:
        blue = 0
    else:
        blue = 138.5177312231 * np.log(tmp_internal - 10) - 305.0447927307

    return np.clip([red, green, blue], 0, 255) / 255


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """Convert sRGB to linear RGB"""
    # scale 0~1
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    """Convert linear RGB to sRGB"""
    # scale 0~1
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * (c ** (1 / 2.4)) - 0.055)


def apply_color_temperature(image: Image.Image, target_temp: float) -> Image.Image:
    """
    Apply a physically-based white balance shift by adjusting the image
    according to relative channel intensity changes between two color temperatures.
    """
    # Get energy scaling at target and reference color temperature
    target_rgb = convert_K_to_RGB(target_temp)
    # Compute scaling factor (how much each channel decays)
    scaling = target_rgb
    scaling = srgb_to_linear(scaling)
    print(f"new scaling: {scaling}")
    
    # Convert image to numpy array and normalize
    img_np = np.asarray(image).astype(np.float32) / 255.0
    img_lin = srgb_to_linear(img_np)

    # Apply energy scaling in linear space
    img_lin[..., 0] *= scaling[0]
    img_lin[..., 1] *= scaling[1]
    img_lin[..., 2] *= scaling[2]

    # Convert back to sRGB
    img_lin = np.clip(img_lin, 0.0, 1.0)
    img_srgb = linear_to_srgb(img_lin)
    img_srgb = np.clip(img_srgb, 0.0, 1.0)

    return Image.fromarray((img_srgb * 255).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser(description="Apply simulated blue light filtering via color temperature transformation.")
    parser.add_argument("--image", "-i", type=str, required=True, help="Path to input image")
    parser.add_argument("--temperature", "-t", type=float, required=True, help="Target color temperature in Kelvin (e.g., 4500)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Path to save the output image")
    args = parser.parse_args()

    # Load and process image
    image = Image.open(args.image).convert("RGB")
    transformed_image = apply_color_temperature(image, target_temp = args.temperature)

    # Set output filename if not specified
    if args.output is None:
        base, ext = os.path.splitext(args.image)
        args.output = f"{base}_transformed_{int(args.temperature)}K{ext}"

    transformed_image.save(args.output)
    print(f"Saved transformed image to: {args.output}")
    

if __name__ == "__main__":
    main()
