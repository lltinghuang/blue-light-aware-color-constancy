import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import gridspec
from PIL import Image


def plot_four_images(image_paths, titles, save_path):
    import os

    import matplotlib.pyplot as plt
    from PIL import Image

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.1, wspace=0.05, top=0.9)  # 預留上方空間給標題

    # 設定大標題
    fig.suptitle("Color Compensation under Low Blue Light Conditions", fontsize=18, weight='bold')

    for ax, img_path, title in zip(axs.ravel(), image_paths, titles):
        img = Image.open(img_path)
        ax.imshow(img)
        ax.set_title(title, fontsize=14)
        ax.axis('off')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_metrics_table(metrics, models, save_path):
    import os

    import matplotlib.pyplot as plt
    import pandas as pd

    # === 整理 metrics → DataFrame ===
    data = {}
    for metric, model_values in metrics.items():
        for model, val in model_values.items():
            if model not in data:
                data[model] = {}
            data[model][metric] = val

    df = pd.DataFrame.from_dict(data, orient="index")
    df.index.name = "Model"
    df = df.reset_index()
    df = df.set_index("Model").reindex(models).reset_index()

    # === 圖片尺寸根據欄數調整 ===
    ncols = len(df.columns)
    nrows = len(df)
    col_width = 1.2  # 每欄寬度（單位調整）
    fig_width = col_width * ncols
    fig_height = 0.6 * nrows + 1.0

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')

    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)

    # 自動設定欄寬根據文字長度
    for i, col in enumerate(df.columns):
        table.auto_set_column_width(i)
    table.scale(1.3, 1.3)  # 放大字體與間距

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main(args):
    df = pd.read_csv(args.csv)

    models = ["original", "pure_transform_2700k", "compensation_once", "compensation_twice"]
    model_titles = models

    grouped = df.groupby(["Subset", "Image"])

    for (subset, image), group in grouped:
        image_paths = [os.path.join(args.image_root, m, subset, image) for m in models]

        if not all(os.path.exists(p) for p in image_paths):
            print(f"[WARN] Missing image(s) for {subset}/{image}, skipping.")
            continue

        metrics = {
            "CIEDE2000": {},
            "Delta_uv_prime": {},
            "Duv": {},
            "EML": {},
            "BlueEnergyPerPixel": {},
        }

        for _, row in group.iterrows():
            model = row["Model"]
            if model in models:
                metrics["CIEDE2000"][model] = f"{row['CIEDE2000']:.2f}"
                metrics["Delta_uv_prime"][model] = f"{row['Delta_uv_prime']:.2f}"
                metrics["Duv"][model] = f"{row['Duv']:.2f}"
                metrics["EML"][model] = f"{row['EML']:.2f}"
                metrics["BlueEnergyPerPixel"][model] = f"{row['BlueEnergyPerPixel']:.4f}"

        base_name = f"{subset}_{os.path.splitext(image)[0]}"
        out_img_path = os.path.join(args.out_dir, f"{base_name}_images.png")
        out_table_path = os.path.join(args.out_dir, f"{base_name}_table.png")

        print(f"[INFO] Processing {base_name} ...")
        plot_four_images(image_paths, model_titles, out_img_path)
        plot_metrics_table(metrics, model_titles, out_table_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="metrics_result.csv", help="Path to metrics CSV file")
    parser.add_argument("--image_root", type=str, default="cp_dataset", help="Path to image root folder")
    parser.add_argument("--out_dir", type=str, default="result_figs", help="Output directory for output images and tables")
    args = parser.parse_args()
    main(args)

# # 跑一張圖的 main
# def main(args):
#     df = pd.read_csv(args.csv)

#     subset = "camera"
#     image = "camera_1.png"
#     models = ["original", "pure_transform_2700k", "compensation_once", "compensation_twice"]
#     model_titles = models

#     group = df[(df["Subset"] == subset) & (df["Image"] == image)]
#     if group.empty:
#         print(f"[ERROR] No data for {subset}/{image}")
#         return

#     image_paths = [os.path.join(args.image_root, m, subset, image) for m in models]
#     if not all(os.path.exists(p) for p in image_paths):
#         print(f"[WARN] Missing image(s) for {subset}/{image}, skipping.")
#         return

#     metrics = {
#         "CIEDE2000": {},
#         "Delta_uv_prime": {},
#         "Duv": {},
#         "EML": {},
#         "BlueEnergyPerPixel": {},
#     }

#     for _, row in group.iterrows():
#         model = row["Model"]
#         if model in models:
#             metrics["CIEDE2000"][model] = f"{row['CIEDE2000']:.2f}"
#             metrics["Delta_uv_prime"][model] = f"{row['Delta_uv_prime']:.2f}"
#             metrics["Duv"][model] = f"{row['Duv']:.2f}"
#             metrics["EML"][model] = f"{row['EML']:.2f}"
#             metrics["BlueEnergyPerPixel"][model] = f"{row['BlueEnergyPerPixel']:.4f}"

#     base_name = f"{subset}_{os.path.splitext(image)[0]}"
#     plot_four_images(image_paths, model_titles, os.path.join(args.out_dir, f"{base_name}_images.png"))
#     plot_metrics_table(metrics, model_titles, os.path.join(args.out_dir, f"{base_name}_table.png"))

