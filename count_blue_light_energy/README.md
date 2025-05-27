# Estimate Image Blue Light Energy

This module estimates the **total** and **average** blue light energy from an input image using RGB-to-SPD conversion and wavelength-based energy integration.

---

## How to Use

```bash
python blue_light_count.py --image <image_path> [--temp 6500] [--precision float64] [--resize 256]

```

###  Arguments

| Argument      | Description                                                                 |
|---------------|-----------------------------------------------------------------------------|
| `--image`     | Path to the input image (e.g., `./images/sample.png`)                      |
| `--temp`      | Color temperature: `2700`, `3500`, `4500`, `5500`, or `6500` (default: 6500)|
| `--precision` | Floating-point precision: `float64` or `float32` (default: `float64`)       |
| `--resize`    | Resize the image’s longer edge to this value. Set `0` to disable resizing. (default: `256`) |

> Resizing is recommended to improve performance with minimal accuracy loss.

---

## Example

```bash
python blue_light_count.py \
  --image ./count_blue_light_energy/images/ori_image.png \
  --temp 6500 \
  --precision float64 \
  --resize 256
```


## Folder Structure

All related code is located in the `count_blue_light_energy/` folder:

```
blue_light_count.py          # Main CLI script
count_blue_light_energy/

├── images/                  # Sample input images
│   └── ori_image.png
├── plots/                   # Estimated SPD plot of hybrid RGB 
│   ├── SPD_plot_pred_255_255_0.png
│   └── ...
├── rgb2spd_lookup           # RGB to SPD lookup table
├── build_rgb2spd_lookup.py  # Build RGB to SPD lookup table
├── plot_rgb2spd.py          # Plot SPD plot of hybrid RGB
├── verify_rgb_spd_gamma.py  # Check if RGB-SPD follows gamma curve (✔️ Yes)
├── verify_rgb_spd_linear.py # Check if RGB-SPD is linear (✘ No)
└── README.md                
```

---

## 📊 Output

After running the script, it will print:

- **Total blue light energy (Counts * nm)**
- **Average blue light energy per pixel (Counts * nm)**

---

