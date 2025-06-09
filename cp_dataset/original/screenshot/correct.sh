#!/bin/bash

for i in $(seq 1 30); do
   pngcrush -ow -rem allb -reduce "screenshot_${i}.png"
done