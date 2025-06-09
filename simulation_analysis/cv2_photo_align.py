# precise_align_visual.py
import argparse
import os

import cv2
import numpy as np


def align_images(img1_path, img2_path, output_path2_aligned, comparison_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # 轉成灰階、用ORB (Oriented FAST and Rotated BRIEF)截取特徵演算法
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    N_MATCHES = 50
    src_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:N_MATCHES]]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:N_MATCHES]]).reshape(-1, 1, 2)

    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    aligned_img2 = cv2.warpAffine(img2, M, (img1.shape[1], img1.shape[0]))

    # 儲存對齊後的 image2
    cv2.imwrite(output_path2_aligned, aligned_img2)
    print(f"Aligned image saved to: {output_path2_aligned}")

    # 建立 side-by-side 比較圖
    comparison = np.hstack((img1, aligned_img2))
    cv2.imwrite(comparison_path, comparison)
    print(f"Comparison image saved to: {comparison_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image1", required=True, help="Reference image (e.g., simulated)")
    parser.add_argument("--image2", required=True, help="To-be-aligned image (e.g., flux)")
    parser.add_argument("--output", default="flux_aligned.jpg", help="Output path for aligned image")
    parser.add_argument("--compare_output", default="comparison.jpg", help="Side-by-side comparison image output")
    args = parser.parse_args()

    align_images(args.image1, args.image2, args.output, args.compare_output)
