import streamlit as st
import numpy as np
import time
from PIL import Image
from scripts.inference import enhance_image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from streamlit_image_comparison import image_comparison

st.set_page_config(page_title="Low Light Image Enhancer", layout="wide")

st.title("🌙 AI Low-Light Image Enhancement System")
st.write("Upload a low-light image and enhance it using deep learning.")

uploaded_file = st.file_uploader(
    "📤 Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    input_image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Original Image")
    st.image(input_image, use_container_width=True)

    if st.button("🚀 Enhance Image"):

        start_time = time.time()

        enhanced_image = enhance_image(input_image)

        processing_time = time.time() - start_time

        st.success("Enhancement Complete!")

        # =============================
        # 🔎 BEFORE / AFTER SLIDER
        # =============================

        st.subheader("🔎 Before vs After Comparison")

        image_comparison(
            img1=input_image,
            img2=enhanced_image,
            label1="Original",
            label2="Enhanced",
            width=700,
        )

        # =============================
        # 📊 IMAGE QUALITY METRICS
        # =============================

        original_np = np.array(input_image)
        enhanced_np = np.array(enhanced_image)

        # Safe Resize Fix
        if original_np.shape != enhanced_np.shape:
            h, w = enhanced_np.shape[:2]
            input_resized = input_image.resize((w, h))
            original_np = np.array(input_resized)

        psnr_value = peak_signal_noise_ratio(original_np, enhanced_np, data_range=255)
        ssim_value = structural_similarity(original_np, enhanced_np, channel_axis=2)

        st.write("### 📊 Image Quality Metrics")

        col1, col2, col3 = st.columns(3)

        col1.metric("PSNR", f"{psnr_value:.2f}")
        col2.metric("SSIM", f"{ssim_value:.4f}")
        col3.metric("Processing Time (s)", f"{processing_time:.2f}")

        # =============================
        # 💾 DOWNLOAD BUTTON
        # =============================

        st.download_button(
            label="📥 Download Enhanced Image",
            data=enhanced_image.tobytes(),
            file_name="enhanced_image.jpg",
            mime="image/jpeg"
        )
