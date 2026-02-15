import streamlit as st
import numpy as np
import cv2
from PIL import Image
from streamlit_image_comparison import image_comparison

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="AI Low Light Enhancer",
    page_icon="🌙",
    layout="wide"
)

# -------------------------
# CUSTOM CSS
# -------------------------
st.markdown("""
<style>
.title {
    text-align: center;
    font-size: 38px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: gray;
    margin-bottom: 30px;
}
.stButton>button {
    width: 100%;
    border-radius: 8px;
    height: 45px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🌙 AI Low Light Image Enhancer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload or Capture an image in real-time</div>', unsafe_allow_html=True)

# -------------------------
# IMAGE ENHANCEMENT FUNCTION
# -------------------------
def enhance_image(image):
    img = np.array(image)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

    return Image.fromarray(enhanced_img)

# -------------------------
# TABS (Professional Layout)
# -------------------------
tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Real-Time Camera"])

# =========================
# TAB 1 — Upload
# =========================
with tab1:
    uploaded_file = st.file_uploader("Upload a low-light image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        enhanced = enhance_image(image)

        image_comparison(
            img1=image,
            img2=enhanced,
            label1="Original",
            label2="Enhanced",
            width=700,
        )

        st.download_button(
            "Download Enhanced Image",
            data=cv2.imencode(".jpg", np.array(enhanced))[1].tobytes(),
            file_name="enhanced_image.jpg",
            mime="image/jpeg"
        )

# =========================
# TAB 2 — Camera
# =========================
with tab2:
    camera_image = st.camera_input("Take a picture")

    if camera_image:
        image = Image.open(camera_image).convert("RGB")
        enhanced = enhance_image(image)

        st.markdown("### 🔍 Live Comparison")

        image_comparison(
            img1=image,
            img2=enhanced,
            label1="Captured",
            label2="Enhanced",
            width=700,
        )

        st.download_button(
            "Download Enhanced Image",
            data=cv2.imencode(".jpg", np.array(enhanced))[1].tobytes(),
            file_name="camera_enhanced.jpg",
            mime="image/jpeg"
        )

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.markdown(
    "<center style='color:gray;'>Developed by Dennis | AI Powered Enhancement System</center>",
    unsafe_allow_html=True
)
