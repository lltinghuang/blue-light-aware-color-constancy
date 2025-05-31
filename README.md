# blue-light-aware-color-constancy
This project uses [Poetry](https://python-poetry.org/) to manage the Python environment and dependencies.

##  Environment Setting
Make sure you have installed *Poetry*.

###  Install Dependencies

From the project root directory, install all required packages without installing the current project as a package:

```bash
poetry install --no-root
```

---

###  Add New Dependencies

To add a new package (e.g., `pandas`), use:

```bash
poetry add pandas
```

---

###  Activate the Virtual Environment

To find the path of the virtual environment, run:

```bash
poetry env info
```

You should see output like this:

```
Virtualenv
Python:         3.10.12
Implementation: CPython
Path:           /home/username/.cache/pypoetry/virtualenvs/blue-light-aware-color-constancy-xxxxxxxx-py3.10
Executable:     /home/username/.cache/pypoetry/virtualenvs/blue-light-aware-color-constancy-xxxxxxxx-py3.10/bin/python
Valid:          True
```

To activate the environment:

```bash
source /home/username/.cache/pypoetry/virtualenvs/blue-light-aware-color-constancy-xxxxxxxx-py3.10/bin/activate
```

> Replace the path with the one shown on your machine.

---

##  Running Scripts

Once activated, you can run your Python scripts as usual:

```bash
python your_script.py
```

Or run them directly via Poetry without activating:

```bash
poetry run python your_script.py
```

## Exit the environment

```bash
deactivate
```

## Before push
please run `pre-commit` to check

## Start to Evaluation

This script allows you to evaluate the perceptual difference between two images using various metrics.

### Supported Metrics

- **SSIM** – Structural Similarity Index  
- **CIE2000** – Delta E (CIEDE2000) color difference  
- **Brightness** – Average brightness in cd/m²  
- **LER** – Luminous Efficacy of Radiation (image-independent)  
- **EML** – Equivalent Melanopic Lux (*to be implemented*)  
- **delta_uv_prime** – Δu'v' color difference (CIE 1976)  
- **Duv** – Distance from the blackbody locus (Ohno 2013)  
- **SSRGB** – Gamut area in a*b* space using convex hull  

> ⚠️ Some metrics (like `SSIM`, `CIE2000`, `delta_uv_prime`, `SSRGB`) require a reference image. Make sure to provide `--ref_image`.

---

### How to Run

```bash
python evaluate.py \
    --image path/to/transformed_image.png \
    --ref_image path/to/reference_image.png \
    --metric SSIM
```

## TODO and Version Updates

### Version 1 (Current)
- Implemented perceptual metrics (excluding EML)
- Used simulated SPD (fake blackbody)
- Tested on simple low-blue-light simulated images

### Planned Updates
- [ ] Add support for **EML (Equivalent Melanopic Lux)** metric
- [ ] Integrate **real SPD profiles** for specific devices/monitors
- [ ] Evaluate performance on **real-world image datasets**
- [ ] Test compatibility with **our proposed color compensation method**

---

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
  python blue_light_count.py \
    --image ./images/sample.png \
    --save_spd --plot_spd
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
