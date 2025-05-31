# Estimate Image Blue Light Energy

This module estimates the **total** and **average** blue light energy from an input image using RGB-to-SPD conversion and wavelength-based energy integration.

---

## How to Use

### Minimal usage

Estimate total and average blue light energy (450–525nm):

```bash
python blue_light_count.py --image ./images/sample.png
```

---

### Additional Parameters

| Argument      | Description                                                                 |
|---------------|-----------------------------------------------------------------------------|
| `--temp`      | Color temperature: `2700`, `3500`, `4500`, `5500`, or `6500` (default: 6500)|
| `--precision` | Floating-point precision: `float64` or `float32` (default: `float64`)       |
| `--resize`    | Resize the image’s longer edge to this value. Set `0` to disable resizing. (default: `256`) |

> Resizing is recommended to improve performance with minimal accuracy loss.

Example with all parameters:

```bash
python blue_light_count.py \
  --image ./count_blue_light_energy/images/ori_image.png \
  --temp 6500 \
  --precision float64 \
  --resize 256
```

---

### Save SPD Output (0531 update)

* #### Save SPD CSV and plot (auto-naming)

  ```bash
  python blue_light_count.py --image ./images/sample.png --save_spd --plot_spd
  ```

  Saves to:

  - `./test_spd_result/sample_spd.csv`
  - `./test_spd_result/sample_spd_plot.png`

* #### Save with custom paths

  ```bash
  python blue_light_count.py \
    --image ./test_image/test_0.5.jpg \
    --save_spd ./test_spd_reulst/test_0.5_spd.csv \
    --plot_spd ./test_spd_reulst/test_0.5_spd_plot.png
  ```

---

## Folder Structure

All related code is located in the `count_blue_light_energy/` folder:

```
blue_light_count.py          # Main CLI script
├── test_image/              # Sample input images
│   ├── test_0.5.jpg
│   └── ...
├── test_spd_result/         # Output folder
│   ├── test_0.5_spd.csv
│   ├── test_0.5_spd_plot.png
│   └── ...

count_blue_light_energy/     # Supplementary utilities
├── images/
├── plots/                   # SPD plots for RGB values
├── rgb2spd_lookup/          # RGB→SPD lookup tables
├── build_rgb2spd_lookup.py
├── plot_rgb2spd.py
├── verify_rgb_spd_gamma.py  # Gamma (fits)
├── verify_rgb_spd_linear.py # inear (not)
└── README.md
```

---

## Output Summary

After running the script:

- Total blue light energy (Counts × nm)
- Average blue light energy per pixel
- *(Optional)* SPD CSV output (0531 update)
- *(Optional)* SPD PNG plot (0531 update)

---