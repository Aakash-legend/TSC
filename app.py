import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# === UI CONFIG (must be first Streamlit command) ===
st.set_page_config(page_title="🌳 Tree Classifier", layout="centered")

# === CLASS NAMES (alphabetically sorted as flow_from_directory uses) ===
# Get class names from dataset folder to match training order
dataset_path = os.path.join(os.path.dirname(__file__), 'Tree_Species_Dataset')
if os.path.exists(dataset_path):
    class_names = sorted([d for d in os.listdir(dataset_path) 
                         if os.path.isdir(os.path.join(dataset_path, d)) and not d.startswith('.')])
else:
    # Fallback if dataset not present (must match training order exactly)
    class_names = [
        "amla", "asopalav", "babul", "bamboo", "banyan", "bili", "cactus", "champa", 
        "coconut", "garmalo", "gulmohor", "gunda", "jamun", "kanchan", "kesudo", "khajur", 
        "mango", "motichanoti", "neem", "nilgiri", "other", "pilikaren", "pipal", 
        "saptaparni", "shirish", "simlo", "sitafal", "sonmahor", "sugarcane", "vad"
    ]

# === LOAD MODEL ===
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'resnet50v2_finetuned_30class_v2.keras')
    return tf.keras.models.load_model(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# === IMAGE PREPROCESSING ===
def preprocess_image(img: Image.Image):
    img = img.convert("RGB")  # Ensure RGB mode
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0,1]
    return np.expand_dims(img_array, axis=0)  # Add batch dim

st.markdown("<h1 style='text-align: center; color: green;'>🌿 Tree Species Classifier</h1>", unsafe_allow_html=True)
st.markdown("Upload an image of a tree to classify it among 30 species.", unsafe_allow_html=True)

# === UPLOAD ===
uploaded_file = st.file_uploader("📷 Upload Tree Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)

    with st.spinner("🔍 Analyzing..."):
        processed = preprocess_image(img)
        preds = model.predict(processed)[0]
        pred_idx = np.argmax(preds)
        confidence = preds[pred_idx] * 100
        pred_class = class_names[pred_idx]

    # === LAYOUT: SIDE BY SIDE ===
    st.markdown("---")
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.image(img, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.markdown(f"<h3>🧠 Predicted Class:</h3><p style='font-size:22px; color:green;'><b>{pred_class}</b></p>", unsafe_allow_html=True)
        st.markdown(f"<h4>🔎 Confidence:</h4><p style='font-size:18px'>{confidence:.2f}%</p>", unsafe_allow_html=True)
        
        # Show top 5 predictions for debugging
        st.markdown("**Top 5 Predictions:**")
        top5_idx = np.argsort(preds)[-5:][::-1]
        for idx in top5_idx:
            st.write(f"- {class_names[idx]}: {preds[idx]*100:.2f}%")