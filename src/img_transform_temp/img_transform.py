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


def apply_color_temperature(image: Image.Image, target_temp: float) -> Image.Image:
    """
    Apply a simulated blue-light filter effect by shifting white balance
    according to the given color temperature.
    """
    r_scale, g_scale, b_scale = convert_K_to_RGB(target_temp)
    img_np = np.asarray(image).astype(np.float32) / 255.0

    img_np[..., 0] *= r_scale
    img_np[..., 1] *= g_scale
    img_np[..., 2] *= b_scale
    img_np = np.clip(img_np, 0, 1)

    return Image.fromarray((img_np * 255).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser(description="Apply simulated blue light filtering via color temperature transformation.")
    parser.add_argument("--image", "-i", type=str, required=True, help="Path to input image")
    parser.add_argument("--temperature", "-t", type=float, required=True, help="Target color temperature in Kelvin (e.g., 4500)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Path to save the output image")
    args = parser.parse_args()

    # Load and process image
    image = Image.open(args.image).convert("RGB")
    transformed_image = apply_color_temperature(image, args.temperature)

    # Set output filename if not specified
    if args.output is None:
        base, ext = os.path.splitext(args.image)
        args.output = f"{base}_transformed_{int(args.temperature)}K{ext}"

    transformed_image.save(args.output)
    print(f"Saved transformed image to: {args.output}")
    

if __name__ == "__main__":
    main()
