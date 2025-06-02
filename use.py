
from spd_utils import extract_avg_spd_from_image

image_path = "./test_image/test_rgb_85_128_85.png"
wl, avg_spd = extract_avg_spd_from_image(image_path, temp=6500, resize_max=256)

# For example, print the results
print("波長 (nm):", wl)
print("平均 SPD 強度:", avg_spd)

# For example, plot the average SPD
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.plot(wl, avg_spd)
plt.title("Average SPD of Image")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Intensity")
plt.grid(True)
plt.show()
# plt.savefig('average_spd_plot.png')
