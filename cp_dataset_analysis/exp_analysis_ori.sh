#!/bin/bash

ORI_ROOT="../cp_dataset/original"
SUBSETS=("camera" "screenshot")
CSV_OUTPUT="./metrics_result.csv"

for subset in "${SUBSETS[@]}"; do
    ORI_DIR="$ORI_ROOT/$subset"

    for ori_path in "$ORI_DIR"/*.{jpg,png}; do
        [ -e "$ori_path" ] || continue
        img_name=$(basename "$ori_path")

        # 避免重複加到 CSV：檢查有沒有 original 記錄
        if grep -q "$subset,$img_name,original" "$CSV_OUTPUT"; then
            echo "[INFO] Already exists: $subset/$img_name"
            continue
        fi

        # 計算 EML 與 BlueEnergyPerPixel
        eml=$(python evaluate.py --image "$ori_path" --metric EML | awk '{print $NF}')
        blue_output=$(python blue_light_count.py --image "$ori_path")
        avg_blue=$(echo "$blue_output" | grep "Average blue energy" | awk '{print $(NF-1)}')

        # 原圖對原圖，所以色差與uv差都是 0
        echo "$subset,$img_name,original,0,0,0,$eml,$avg_blue" >> "$CSV_OUTPUT"
        echo "$subset/$img_name → original metrics"
    done
done
