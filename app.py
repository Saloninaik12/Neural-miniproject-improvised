import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Import helper functions
from utils import load_model, preprocess_for_model, decode_predictions, make_gradcam_heatmap

st.set_page_config(page_title="Multi-Model Image Classifier", page_icon="🔍", layout="centered")

# ----------------------- UI Header -----------------------
st.title("🖼️ Multi-Model Image Classifier")
st.write("Upload an image and choose a model to analyze what the object is!")

# ----------------------- Model Selection -----------------------
model_name = st.selectbox(
    "Choose a model:",
    ["ResNet50", "VGG16", "MobileNet"]  # must match utils.py
)

st.info(f"Selected Model: **{model_name}**")

# Load model once with caching
@st.cache_resource
def load_selected_model(name):
    return load_model(name)

model, preprocess_fn, decode_fn = load_selected_model(model_name)

# ----------------------- File Uploader -----------------------
uploaded_file = st.file_uploader("📁 Upload an image...", type=["jpg", "jpeg", "png"])

# ----------------------- Prediction Section -----------------------
if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Predict button
    if st.button("🔍 Classify Image"):
        
        # Preprocess
        arr = preprocess_for_model(img, preprocess_fn)

        # Predict
        preds = model.predict(arr)
        decoded = decode_fn(preds, top=5)

        st.subheader("✅ Top 5 Predictions:")
        for (_, label, prob) in decoded[0]:
            st.write(f"**{label}** — {prob * 100:.2f}%")

        # ------------------ Grad-CAM Heatmap ------------------
        st.subheader("🔥 Grad-CAM (Model Attention Heatmap)")

        # Default last conv layer for each model
        last_layer = {
            "ResNet50": "conv5_block3_out",
            "VGG16": "block5_conv3",
            "MobileNet": "conv_pw_13_relu"
        }[model_name]

        heatmap = make_gradcam_heatmap(arr, model, last_layer)

        # -------------- Overlay Heatmap WITHOUT cv2 --------------
        heatmap_img = Image.fromarray(np.uint8(255 * heatmap))
        heatmap_img = heatmap_img.resize(img.size)

        # Create transparent color map
        heatmap_rgba = Image.new("RGBA", heatmap_img.size)
        heatmap_rgba.putalpha(heatmap_img)

        # Overlay
        overlay = Image.alpha_composite(img.convert("RGBA"), heatmap_rgba)

        st.image(overlay, caption="Grad-CAM Heatmap", use_column_width=True)

    if st.button("🔁 Try another image"):
        st.experimental_rerun()

else:
    st.info("Please upload an image to begin.")
