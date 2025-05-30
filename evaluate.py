import argparse
import cv2
import numpy as np
import sys
from scipy.spatial import ConvexHull
from skimage.metrics import structural_similarity as compare_ssim
import colour  # install colour-science
from colour.difference import delta_E_CIE2000
from colour.colorimetry import sd_to_XYZ, MSDS_CMFS


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def compute_brightness(image, display_max_luminance=100):
    XYZ = colour.sRGB_to_XYZ(image / 255.0)
    Y = XYZ[..., 1]  # Luminance channel ([0, 1])
    brightness = np.mean(Y) * display_max_luminance
    return brightness

# LER have no relation with image!
def compute_ler(spd=None, temperature=6500):
    if spd is None:
        spd = colour.sd_blackbody(temperature, colour.SpectralShape(360, 830, 1)) # Blackbody SPD at 6500K
    return colour.luminous_efficacy(spd)

def sd_to_illuminance(sd, cmfs=MSDS_CMFS['CIE 1931 2 Degree Standard Observer']):
    XYZ = sd_to_XYZ(sd, cmfs=cmfs)
    Y = XYZ[1]
    k = 683  # lm/W for photopic vision
    return Y * k

# def compute_eml(image, spd=None, temperature=6500):
#     if spd is None:
#         # Use blackbody approximation from temperature (in Kelvin)
#         spd = colour.sd_blackbody(temperature, colour.SpectralShape(360, 830, 1))

#     # Normalize SPD based on image brightness
#     # Scale so the photopic lux matches the mean brightness of the image
#     photopic_lux_target = 683 * image.mean()
#     photopic_lux_original = sd_to_illuminance(spd)
#     scale_factor = photopic_lux_target / photopic_lux_original
#     spd *= scale_factor

#     # Load melanopic sensitivity curve
#     melanopic_sd = colour.SDS_TO_XYZ['melanopic']

#     # Compute photopic and melanopic lux
#     melanopic_lux = colour.sd_to_illuminance(spd, relative_spd=melanopic_sd)

#     # EML = melanopic_lux (after normalization to match image luminance)
#     return melanopic_lux

def XYZ_to_uv_prime(XYZ):
    """
    Convert CIE XYZ to CIE 1976 u'v' chromaticity coordinates.
    """
    X, Y, Z = np.moveaxis(XYZ, -1, 0)
    denominator = X + 15 * Y + 3 * Z + 1e-10  # Avoid divide by zero
    u_prime = (4 * X) / denominator
    v_prime = (9 * Y) / denominator
    return u_prime, v_prime

def compute_uv_prime(image):
    """
    Compute average CIE 1976 u'v' coordinates from an sRGB image.
    """
    XYZ = colour.sRGB_to_XYZ(image.astype(np.float32) / 255.0)
    u_prime, v_prime = XYZ_to_uv_prime(XYZ)
    return np.mean(u_prime), np.mean(v_prime)

def compute_delta_uv_prime(image1, image2):
    """
    Compute Δu'v' between two images.
    """
    u1, v1 = compute_uv_prime(image1)
    u2, v2 = compute_uv_prime(image2)
    return np.sqrt((u2 - u1) ** 2 + (v2 - v1) ** 2)

def compute_duv(image):
    u_prime, v_prime = compute_uv_prime(image)
    u = u_prime
    v = 2 * v_prime / 3
    _, Duv = colour.uv_to_CCT((u, v), method='Ohno 2013', return_D_uv=True)
    return Duv

def compute_ssrgb(image):
    # Convert RGB [0–255] to float [0–1]
    flat_rgb = image.reshape(-1, 3).astype(np.float32) / 255.0

    # Convert to XYZ then Lab
    xyz = colour.sRGB_to_XYZ(flat_rgb)
    lab = colour.XYZ_to_Lab(xyz)

    # Only use a*, b* for gamut estimation
    ab = lab[:, 1:3]

    # Compute convex hull area in ab plane
    hull = ConvexHull(ab)
    return hull.area

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_image', help='Path to the reference image')
    parser.add_argument('--image', help='Path to the transformed image')
    parser.add_argument('--metric', required=True, choices=['SSIM', 'CIE2000', 'Brightness', 'LER', 'EML', 'delta_uv_prime', 'Duv', 'SSRGB'], help='Metric to use for evaluation')
    args = parser.parse_args()

    metrics_require_ref = ['SSIM', 'CIE2000', 'delta_uv_prime', 'SSRGB']
    if args.metric in metrics_require_ref and not args.ref_image:
        print(f"Error: --ref_image is required for metric {args.metric}")
        sys.exit(1)

    if args.metric in metrics_require_ref:
        ref = load_image(args.ref_image)
    img = load_image(args.image)

    if args.metric == 'SSIM':
        score, _ = compare_ssim(ref, img, channel_axis=-1, full=True, win_size=7)

    elif args.metric == 'CIE2000':
        score = np.mean(delta_E_CIE2000(ref, img))

    elif args.metric == 'Brightness':
        score = compute_brightness(img)

    elif args.metric == 'LER':
        score = compute_ler()

    # elif args.metric == 'EML':
    #     score = compute_eml(img)

    elif args.metric == 'delta_uv_prime':
        score = compute_delta_uv_prime(ref, img)

    elif args.metric == 'Duv':
        score = compute_duv(img)

    elif args.metric == 'SSRGB':
        gamut_ref = compute_ssrgb(ref)
        gamut_trans = compute_ssrgb(img)
        score = gamut_trans / gamut_ref  # Ratio of transformed to reference gamut area

    else:
        raise ValueError(f"Unsupported metric: {args.metric}")

    print(f"{args.metric} score: {score:.4f}")


if __name__ == '__main__':
    main()
