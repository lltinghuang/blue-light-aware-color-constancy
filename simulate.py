import cv2
import numpy as np
from matplotlib import pyplot as plt

def simulate_low_blue_light(img_path, blue_factor=0.5, red_boost=1.05, green_boost=1.0):
    """
    Simulate low blue light effect on an image.
    :param img: input image (numpy array)
    :param blue_factor: blue light reduction factor (0-1)
    :param red_boost: red boost factor (1.0 means no change)
    :param green_boost: green boost factor (1.0 means no change)
    :return: 模擬後的圖片 (numpy array)
    """
    img_bgr = cv2.imread(input_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Not found {input_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0  # normalize to 0-1

    low_blue = img_rgb.copy()
    low_blue[..., 0] *= red_boost     # R channel
    low_blue[..., 1] *= green_boost   # G channel
    low_blue[..., 2] *= blue_factor   # B channel

    # clip values to [0, 1], then convert back to uint8
    low_blue = np.clip(low_blue, 0, 1)
    low_blue_uint8 = (low_blue * 255).astype(np.uint8)
    img_rgb_uint8 = (img_rgb * 255).astype(np.uint8)

    # save simulated image
    output_path = f"low_blue_light_simulated_{blue_factor}.jpg"
    low_blue_bgr = cv2.cvtColor(low_blue_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, low_blue_bgr)
    print(f"The simulated image has been saved as '{output_path}'")

def show_comparison(original, simulated):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(simulated)
    plt.title("Low Blue Light Simulated")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    input_path = "images.jpg"
    simulate_low_blue_light(input_path, blue_factor=0.5, red_boost=1.05, green_boost=1.0)
    simulate_low_blue_light(input_path, blue_factor=0.6, red_boost=1.05, green_boost=1.0)
    simulate_low_blue_light(input_path, blue_factor=0.7, red_boost=1.05, green_boost=1.0)
    simulate_low_blue_light(input_path, blue_factor=0.8, red_boost=1.05, green_boost=1.0)
    simulate_low_blue_light(input_path, blue_factor=0.9, red_boost=1.05, green_boost=1.0)
    
