# Simulated Blue Light Filtering

This Python module simulates **blue light filtering** by transforming an image’s appearance to match a specified **display color temperature** (in Kelvin), mimicking the effects of software like **f.lux** or **Night Shift**.

The system uses the **CIECAM02 color appearance model** to simulate how an image would appear under a different illuminant (e.g., 2700K) and then applies an **inverse transformation** to adjust the image so that, when viewed on a standard display (usually D65), it appears as if it were rendered under the target warmer light.

---

## Features

* **Color Appearance Modeling** with `colour-science` (CIECAM02)
* **Support for arbitrary color temperatures** from 1000K to 25000K
* **Inverse transformation** to pre-compensate for target ambient color temperature
* Based on RGB-to-XYZ conversions, chromatic adaptation, and perceptual rendering

---

## Example Usage

```bash
python ciecam_transform.py \
 --img ../../img_transform_temp/img/cute_furret_small.png \
 --temp 2700 \
 --out cute_furret_2700_expected.png \
 --out_optimized cute_furret_2700_optimized.png
```

```bash=
python ciecam_transform.py --img ../../img_transform_temp/img/cute_furret_small.png
```

---

## Arguments

| Argument          | Description                                                                                |
| ----------------- | ------------------------------------------------------------------------------------------ |
| `--img`           | Path to the input image (e.g., `./images/photo.jpg`)                                       |
| `--temp`          | Target ambient display color temperature (Kelvin), e.g., `2700`, `3500`                    |
| `--out`           | Output filename for the perceptually simulated image (default: `Temperature_expected.png`) |
| `--out_optimized` | Output filename for the inverse-transformed image (default: `Program_optimized.png`)       |

> The script first computes the perceptual appearance of an image under the target light (e.g., 2700K), then applies an inverse linear RGB scaling so that the resulting image, when viewed under normal lighting, appears similar to the warmer-light version.

---

## Output

1. **Perceptual Simulation** (`--out`):
   How the image would *look* under warmer (low blue light) illumination.

2. **Inverse-Transformed Image** (`--out_optimized`):
   An adjusted image that, when displayed under normal lighting, *resembles* the simulation result.

---
