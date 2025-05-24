## Experiment Data from 5/24

### Introduction

This dataset includes measurements from three different devices. Among them, the **Huawei Mate** device contains a complete dataset that covers both RGB value variation and color temperature changes. The **MacBook Pro** and **ASUS TUF** datasets, in contrast, only include measurements related to color temperature changes.

---

### Important Notes (for MacBook Pro and ASUS TUF datasets)

For both the MacBook Pro and ASUS TUF data, the background was initially set as “natural light,” which is not ideal. To correct this, we collected an additional reference measurement named **Natural\_Light\_Bias**, where we used a completely black screen (RGB = 0,0,0) to capture the ambient lighting condition.

**To interpret the data correctly**, you must apply the bias correction as follows:

```python
corrected_data = measured_data + natural_light_bias
```
* 注: 這我真的也不確定 怎麼樣看起來都怪怪的，因為我用華為得dataset來當作參考，怎樣都覺得資料對不起來? 反正我多提供了一個 natural light bias 當參考，我也不確定這東西有沒有用，但我覺得應該還好，因為我看那些數值都是大幾萬再跑的，我覺得還是可以先用原本的dataset跑起來，如果data 不合理再把bias加上去看會不會比較好?

---

### Huawei Dataset

In addition to temperature-based color measurements, the Huawei dataset also includes detailed intensity data for a wide range of **RGB values**. These measurements were taken under a fixed temperature of **6500K**, with all blue light filtering features **disabled**.

File naming convention:
Each folder is named based on the RGB value it represents. For example, a folder named `85_255_128` corresponds to a color setting of (R=85, G=255, B=128). The “Hybrid\_Color” folder contains such mixed RGB samples.

---

### How to Use the Dataset

1. **Analyze pure RGB spectra (Huawei)**
   Examine the spectral intensity distributions of pure RGB values—such as (255,0,0), (0,255,0), (0,0,255)—to understand how intensity varies across different wavelengths.

2. **Evaluate intensity decay in fixed channels**
   In the `Fix_Channel` folder, analyze how intensity changes with decreasing RGB values. This can help model the tone curve or gamma transformation used by the display.

3. **Study hybrid color behavior**
   Based on the spectral data from mixed RGB values, propose a model (e.g., linear or weighted combination of pure RGB spectra) and validate it against actual measurements.

4. **Quantify temperature effects**
   Investigate how changes in color temperature affect the spectral intensity across different wavelengths. With the above RGB-intensity relationships, you can estimate the spectral output at various color temperatures and RGB settings.
