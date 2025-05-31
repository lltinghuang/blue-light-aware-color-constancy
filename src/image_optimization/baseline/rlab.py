import argparse

import colour  # install colour-science
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def find_unique_colors(img_path):
    img = Image.open(img_path).convert('RGB')
    img_np = np.array(img)
    pixels = img_np.reshape(-1, 3)
    unique_colors = np.unique(pixels, axis=0)
    return unique_colors

def plot_xyz_colors(colors_rgb):
    # Convert RGB [0,255] to [0,1] for colour-science
    rgb_normalized = colors_rgb / 255.0
    xyz_colors = np.array([colour.sRGB_to_XYZ(rgb) for rgb in rgb_normalized])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs = xyz_colors[:, 0]
    ys = xyz_colors[:, 1]
    zs = xyz_colors[:, 2]

    ax.scatter(xs, ys, zs, c=rgb_normalized, marker='o', s=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Distinct Colors in XYZ Color Space')
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, required=True, help='Path to input image')
    args = parser.parse_args()

    unique_colors = find_unique_colors(args.img)
    print(f"Found {len(unique_colors)} unique RGB values.")
    plot_xyz_colors(unique_colors)

if __name__ == '__main__':
    main()
