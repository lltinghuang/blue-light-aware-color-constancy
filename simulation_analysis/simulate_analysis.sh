#!/bin/bash

SIM_IMG="../test_image/simulated_2700K.jpg"
FLUX_IMG="../test_image/flux_2700K.jpg"
ALIGNED_IMG="../test_image/flux_2700K_aligned.jpg"

# 使用opencv對齊兩張拍攝的照片
echo "使用opencv對齊兩張拍攝的照片"
python cv2_photo_align.py \
    --image1 $SIM_IMG \
    --image2 $FLUX_IMG \
    --output $ALIGNED_IMG \
    --compare_output "$COMPARE_IMG"
echo ""

# Part1: 比較兩張圖的matrix
echo "Compare the simulated image & flux image"
# 1. SSIM 相似度
echo -e "\n1. SSIM 結構相似性"
python evaluate.py --image $SIM_IMG --ref_image $ALIGNED_IMG --metric SSIM

# 2. CIEDE2000 色差
echo -e "\n2. CIE2000 色差"
python evaluate.py --image $SIM_IMG --ref_image $ALIGNED_IMG --metric CIE2000

# 3. Δu'v' 色差 (CIE 1976)
echo -e "\n3. Δu'v' 色差（CIE 1976）"
python evaluate.py --image $SIM_IMG --ref_image $ALIGNED_IMG --metric delta_uv_prime

# 4. 色域面積比較 SSRGB – Gamut area in ab space using convex hull
echo -e "\n4. SSRGB (gamut are)"
python evaluate.py --image $SIM_IMG --ref_image $ALIGNED_IMG --metric SSRGB

# Part2-1: 計算一張圖的matrix - simulated image
# echo "\nCalculate metrics for the simulated image"
# # a. Brightness: Average brightness in cd/m², based on the physical luminance (Y channel in XYZ).
# echo "\na. Brightness(Average brightness in cd/m²)"
# python evaluate.py --image $SIM_IMG --metric Brightness

# # b. Lightness: Perceptual lightness (L*), computed from Y relative to a reference white point; reflects human visual sensitivity to brightness.
# echo "\nb. Lightness"
# python evaluate.py --image $SIM_IMG --metric Lightness

# c. Duv – Distance from the blackbody locus (Ohno 2013)
echo -e "\nc. Duv "
python evaluate.py --image $SIM_IMG --metric Duv

# d. Blue Light Energy: 
echo -e "\nd. Blue Light Energy"
python blue_light_count.py --image $SIM_IMG

# Part2-2: 計算一張圖的matrix - flux image
# echo "\nCalculate metrics for the simulated image"
# # a. Brightness: Average brightness in cd/m², based on the physical luminance (Y channel in XYZ).
# echo "\na. Brightness(Average brightness in cd/m²)"
# python evaluate.py --image $ALIGNED_IMG --metric Brightness

# # b. Lightness: Perceptual lightness (L*), computed from Y relative to a reference white point; reflects human visual sensitivity to brightness.
# echo "\nb. Lightness"
# python evaluate.py --image $ALIGNED_IMG --metric Lightness

# c. Duv – Distance from the blackbody locus (Ohno 2013)
echo -e "\nc. Duv "
python evaluate.py --image $ALIGNED_IMG --metric Duv

# d. Blue Light Energy: 
echo -e "\nd. Blue Light Energy"
python blue_light_count.py --image $ALIGNED_IMG




