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
    ["ResNet50", "VGG16", "MobileNetV2"]
)

st.info(f"Selected Model: **{model_name}**")

# Load model once with caching
@st.cache_resource
def load_selected_model(name):
    return load_model(name)

model = load_selected_model(model_name)

# ----------------------- File Uploader -----------------------
uploaded_file = st.file_uploader("📁 Upload an image...", type=["jpg", "jpeg", "png"])

# ----------------------- Prediction Section -----------------------
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Predict button
    if st.button("🔍 Classify Image"):
        
        # Preprocess
        arr = preprocess_for_model(img, model_name)

        # Predict
        preds = model.predict(arr)
        decoded = decode_predictions(preds, model_name)

        st.subheader("✅ Top 5 Predictions:")
        for (_, label, prob) in decoded:
            st.write(f"**{label}** — {prob * 100:.2f}%")

        # ------------------ Grad-CAM Heatmap ------------------
        st.subheader("🔥 Grad-CAM (Model Attention Heatmap)")

        heatmap = make_gradcam_heatmap(arr, model)

        # Convert to image and overlay
        import cv2
        img_array = np.array(img.resize((224, 224)))
        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_array, 0.6, heatmap_color, 0.4, 0)

        st.image(overlay, caption="Grad-CAM Heatmap", use_column_width=True)

    if st.button("🔁 Try another image"):
        st.experimental_rerun()
else:
    st.info("Please upload an image to begin.")
