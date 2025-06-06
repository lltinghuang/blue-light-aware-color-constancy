# Simulated Blue Light Filtering

This module simulates **blue light reduction** by transforming an image’s white balance to a given **color temperature** (in Kelvin), similar to f.lux or Night Shift.

---

## How to Use

```bash
python img_transform.py -i ./img/cute_furret.jpg -t 2700 -o cute_furret_2700.png
```

### Arguments

| Argument              | Description                                                                |
| --------------------- | -------------------------------------------------------------------------- |
| `--image`, `-i`       | Path to the input image (e.g., `./img/cute_furret.jpg`)                    |
| `--temperature`, `-t` | Target color temperature in Kelvin (e.g., `4500`, `3500`, `2700`)          |
| `--output`, `-o`      | Path to save the transformed image. If not set, auto-generates a filename. |

> The transformation approximates a chromatic adaptation using an RGB scaling factor derived from the target color temperature.

---

### TODO

Quantitative analysis
