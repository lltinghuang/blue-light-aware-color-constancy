#!/bin/bash

# Set temperature value
TEMP=2700

# Process camera_1.png to camera_30.png
for i in $(seq 1 30); do
    INPUT="../cp_dataset/original/camera/camera_${i}.png"
    OUTPUT="../cp_dataset/pure_transform_2700k/camera/camera_${i}.png"
    python img_transform.py -i "$INPUT" --temp "$TEMP" --o "$OUTPUT"
done

# Process screenshot_1.png to screenshot_18.png
for i in $(seq 1 18); do
    INPUT="../cp_dataset/original/screenshot/screenshot_${i}.png"
    OUTPUT="../cp_dataset/pure_transform_2700k/screenshot/screenshot_${i}.png"
    python img_transform.py -i "$INPUT" --temp "$TEMP" --o "$OUTPUT"
done
