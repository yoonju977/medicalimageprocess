import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image
from skimage import feature
from tqdm import tqdm

st.set_page_config(layout="wide", page_title="Team 1-Project:")

st.sidebar.write("## upper image")
upperImg = st.sidebar.file_uploader("Upload upper image", type=[".img"])

st.sidebar.write("## lower image")
lowerImg = st.sidebar.file_uploader("Upload lower image", type=[".img"])


st.sidebar.write("## Registration Parameter")

param_x = st.sidebar.number_input("x", value=0.0, format="%.10f")
# st.write(f"x = {param_x}")

param_y = st.sidebar.number_input("y", value=0.0, format="%.10f")
# st.write(f"y = {param_y}")

Lambda = st.sidebar.number_input(
    "λ", value=0.0, min_value=0.0, max_value=1.0, format="%.10f"
)
# st.write(f"λ = {Lambda}")


upload = [upperImg, lowerImg, param_x, param_y, Lambda]


# Function to load .img file as a 2D numpy array (1 channel)
def load_img_file(file, shape=(3072, 3072), dtype=np.uint16):
    if file:
        # Read binary data
        data = np.frombuffer(file.getbuffer(), dtype=dtype)
        # Reshape into 2D array (1 channel)
        try:
            volume = data.reshape(shape)
            return volume
        except ValueError:
            st.error("The file size does not match the specified dimensions!")
            return None
    return None


def align_image(lower_image, x, y):
    nr, nc = lower_image.shape
    fft = np.fft.fft2(lower_image)
    Nr = np.fft.ifftshift(np.arange(-np.fix(nr // 2), np.ceil(nr // 2)))
    Nc = np.fft.ifftshift(np.arange(-np.fix(nc // 2), np.ceil(nc // 2)))
    Nc, Nr = np.meshgrid(Nc, Nr)
    phase_shift = np.exp(1j * 2 * np.pi * (x * Nr / nr + y * Nc / nc))
    shifted_image = np.fft.ifft2(fft * phase_shift)
    return np.real(shifted_image)


def convex_combine_images(upper_image, lower_image, lbd):
    if upper_image.shape != lower_image.shape:
        raise ValueError("Images must have the same shape for combination.")
    return lbd * upper_image + (1 - lbd) * lower_image


def scale_image(image):
    normalized_data = (image - image.min()) / (image.max() - image.min())
    return (normalized_data * 65535).astype(np.uint16)


upper_data = load_img_file(upperImg) if upperImg else None
lower_data = load_img_file(lowerImg) if lowerImg else None

if upper_data is not None and lower_data is not None:
    # Step 1 컨테이너
    step1_container = st.container()
    step1_container.markdown("### Step 1: Aligning Images with Fourier Transform...")
    step1_progress_bar = step1_container.progress(0)
    col_s1_left, col_s1_right = step1_container.columns(2)

    with st.spinner("Aligning Lower Image..."):
        aligned_lower = align_image(lower_data, param_x, param_y)
        time.sleep(1)  # Simulate processing delay
        step1_progress_bar.progress(100)
        step1_container.markdown("**Step 1 Complete!**")

    # Step 2 컨테이너
    step2_container = st.container()
    step2_container.markdown("### Step 2: Combining Images...")
    step2_progress_bar = step2_container.progress(0)
    col_s2_left, col_s2_right = step2_container.columns(2)

    with st.spinner("Combining Images..."):
        combined_image = convex_combine_images(upper_data, aligned_lower, Lambda)
        time.sleep(2)  # Simulate processing delay
        step2_progress_bar.progress(100)
        step2_container.markdown("**Step 2 Complete!**")

    # Step 3 컨테이너
    step3_container = st.container()
    step3_container.markdown("### Step 3: Scaling Final Image...")
    step3_progress_bar = step3_container.progress(0)
    col_s3_left, col_s3_right = step3_container.columns(2)

    with st.spinner("Scaling Final Image..."):
        final_image = scale_image(combined_image)
        time.sleep(3)  # Simulate processing delay
        step3_progress_bar.progress(100)
        step3_container.markdown("**Step 3 Complete!**")

    st.success("Processing Complete!")

    # 최종 결과를 로컬에 저장 (img 포맷 - raw binary)
    output_path = r"output/convex_comb_low_dose_3.img"
    with open(output_path, "wb") as f:
        final_image.tofile(f)
    st.write(f"Final image saved locally as {output_path}.")

else:
    st.warning("Please upload both upper and lower images to proceed.")
