#!/bin/bash

# 原圖與壓縮圖資料夾路徑
ORI_ROOT="../cp_dataset/original"
COM_ROOTS=("../cp_dataset/compensation_once" "../cp_dataset/compensation_twice" "../cp_dataset/pure_transform_2700k")
# MODEL_TAGS=("COM_0" "COM_1" "COM_2")

# 子資料夾類型
SUBSETS=("camera" "screenshot")

# output csv
CSV_OUTPUT="./metrics_result.csv"
echo "Subset,Image,Model,CIEDE2000,Delta_uv_prime,Duv,EML,BlueEnergyPerPixel" > "$CSV_OUTPUT"

for subset in "${SUBSETS[@]}"; do
    ORI_DIR="$ORI_ROOT/$subset"

    for ori_path in "$ORI_DIR"/*.{jpg,png}; do
        [ -e "$ori_path" ] || continue
        img_name=$(basename "$ori_path")

        for comp_root in "${COM_ROOTS[@]}"; do
            COMP_IMG="${comp_root}/${subset}/${img_name}"
            [ -f "$COMP_IMG" ] || { echo "[WARN] File not found: $COMP_IMG" >&2; continue; }

            model_name=$(basename "$comp_root")

            cie2000=$(python evaluate.py --image "$COMP_IMG" --ref_image "$ori_path" --metric CIEDE2000 | awk '{print $NF}')
            delta_uv=$(python evaluate.py --image "$COMP_IMG" --ref_image "$ori_path" --metric delta_uv_prime | awk '{print $NF}')
            duv=$(python evaluate.py --image "$COMP_IMG" --metric Duv | awk '{print $NF}')
            eml=$(python evaluate.py --image "$COMP_IMG" --metric EML | awk '{print $NF}')

            blue_output=$(python blue_light_count.py --image "$COMP_IMG")
            avg_blue=$(echo "$blue_output" | grep "Average blue energy" | awk '{print $(NF-1)}')

            echo "$subset,$img_name,$model_name,$cie2000,$delta_uv,$duv,$eml,$avg_blue" >> "$CSV_OUTPUT"
        done
    done
done
