import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
from colour import (CCS_ILLUMINANTS, CIECAM02_to_XYZ, XYZ_to_CIECAM02,
                    XYZ_to_sRGB, sRGB_to_XYZ, xy_to_xyY, xyY_to_XYZ)
from PIL import Image


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_image(rgb_image, path):
    rgb_uint8 = np.clip(rgb_image * 255, 0, 255).astype(np.uint8)
    Image.fromarray(rgb_uint8).save(path)

def simulate_d27_CIECAM02(rgb_image, Y_n=20, surround='Average'):
    rgb_image = rgb_image / 255.0
    xyz = np.array([sRGB_to_XYZ(rgb) * 100 for rgb in rgb_image.reshape(-1, 3)])

    # White points
    XYZ_w_d65 = xyY_to_XYZ(xy_to_xyY(CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]))
    # LED-B1	0.4560	0.4560 (wiki)
    XYZ_w_d27 = xyY_to_XYZ(xy_to_xyY(np.array([0.54369557, 0.32107944])))

    from colour.appearance import VIEWING_CONDITIONS_CIECAM02
    vc = VIEWING_CONDITIONS_CIECAM02[surround]
    Y_b = 20.0        # background luminance
    vc = VIEWING_CONDITIONS_CIECAM02[surround]
    # Original appearance under D65
    # Compute appearance under D65
    appearance_d65 = [
        XYZ_to_CIECAM02(x, XYZ_w_d65, Y_n, Y_b, surround=vc)
        for x in xyz
    ]

    # Adapt appearance to D27 and convert back to XYZ
    xyz_d27 = [
        CIECAM02_to_XYZ(a, XYZ_w_d27, Y_n, Y_b, surround=vc)
        for a in appearance_d65
    ]

    # Back to sRGB
    rgb_d27 = np.stack([XYZ_to_sRGB(x / 100) for x in xyz_d27])
    return np.clip(rgb_d27.reshape(rgb_image.shape), 0, 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', required=True, help='Input image path')
    parser.add_argument('--out', default='d27_output.png', help='Output path for D27 image')
    parser.add_argument('--clusters', type=int, default=1000, help='Number of clusters for visualization')
    args = parser.parse_args()

    # Load image
    print(f"Loading image: {args.img}")
    img = load_image(args.img)

    # Simulate D27 appearance
    print("Simulating perceptual appearance under D27...")
    img_d27 = simulate_d27_CIECAM02(img)
    save_image(img_d27, args.out)
    print(f"Saved simulated image to {args.out}")


if __name__ == '__main__':
    main()
