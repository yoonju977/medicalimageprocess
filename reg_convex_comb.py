import os
import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt

orig_upper = "Image_final/low_dose/upper/Chest_PA_Human.IMG"
orig_lower = "Image_final/low_dose/lower/Chest_PA_Human.IMG"

width, height = 3072, 3072
dtype = np.uint16

# .IMG 파일 읽기
with open(orig_upper, 'rb') as f :
    orig_u_raw_data = np.fromfile(f, dtype = dtype)

# 데이터 배열을 2D 이미지로 변환
orig_upper = orig_u_raw_data.reshape((height, width))

with open(orig_lower, 'rb') as f :
    orig_l_raw_data = np.fromfile(f, dtype = dtype)

# 데이터 배열을 2D 이미지로 변환
orig_lower = orig_l_raw_data.reshape((height, width))

x = 0.481006
y = -7.629161

nr, nc = orig_lower.shape
fft = np.fft.fft2(orig_lower)

Nr = np.fft.ifftshift(np.arange(-np.fix(nr // 2), np.ceil(nr // 2)))
Nc = np.fft.ifftshift(np.arange(-np.fix(nc // 2), np.ceil(nc // 2)))
Nc, Nr = np.meshgrid(Nc, Nr)

phase_shift = np.exp(1j * 2 * np.pi * (x * Nr / nr + y * Nc / nc))
shifted_image = np.fft.ifft2(fft * phase_shift) * np.exp(-1j * 0)

final_lower = np.real(shifted_image)


def convex_combine_images(upper_image, lower_image, lbd):
    if upper_image.shape != lower_image.shape :
        raise ValueError("Images must have the same shape for convex combination.")
    combined_image = lbd * upper_image + (1 - lbd) * lower_image
    
    return combined_image

final_orig = convex_combine_images(orig_upper, final_lower, 0.6)


normalized_data = (final_orig - final_orig.min()) / (final_orig.max() - final_orig.min())

# 2. uint16 범위로 스케일링 (0~65535)
uint16_data = (normalized_data * 65535).astype(np.uint16)

output_path = r"output/convex_comb_low_dose_3.img"

with open(output_path, "wb") as f:
    uint16_data.tofile(f)
