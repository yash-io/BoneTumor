import streamlit as st
import torch
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from torchvision import models, transforms
import os
import gdown

# ================= SETTINGS =================
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

IMG_SIZE = 224
THRESHOLD = 0.001
DEVICE = "cpu"

# ================= DOWNLOAD MODELS =================
os.makedirs("models", exist_ok=True)

DENSENET_PATH = "models/best_densenet_btxrd.pth"
UNET_PATH = "models/best_unet_btxrd.h5"

# ✅ FIXED LINKS
DENSENET_URL = "https://drive.google.com/uc?id=1gGNsPyeDb-oLQ0K14HFNYpWW4auohWzg"
UNET_URL = "https://drive.google.com/uc?id=1cKucZBoFr5sL6VQoc3YawSri69nvwuPp"

if not os.path.exists(DENSENET_PATH):
    st.write("Downloading DenseNet model...")
    gdown.download(DENSENET_URL, DENSENET_PATH, quiet=False)

if not os.path.exists(UNET_PATH):
    st.write("Downloading U-Net model...")
    gdown.download(UNET_URL, UNET_PATH, quiet=False)

# ================= LOAD MODELS =================
@st.cache_resource
def load_models():
    # DenseNet
    model_dense = models.densenet121()
    model_dense.classifier = torch.nn.Linear(model_dense.classifier.in_features, 2)
    model_dense.load_state_dict(torch.load(DENSENET_PATH, map_location=DEVICE))
    model_dense.eval()

    # U-Net
    model_unet = tf.keras.models.load_model(UNET_PATH, compile=False)

    return model_dense, model_unet

model_dense, model_unet = load_models()

# ================= TRANSFORMS =================
dense_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ================= CLASSIFICATION =================
def classify_image(image):
    img = image.convert("RGB").resize((224, 224))
    img = dense_transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model_dense(img)
        probs = torch.softmax(output, dim=1)[0]

    return float(probs[1])

# ================= SEGMENTATION =================
def segment_image(image):
    img = np.array(image)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    input_arr = np.expand_dims(img_resized / 255.0, axis=0)

    pred_mask = model_unet.predict(input_arr)[0].squeeze()
    pred_mask_bin = (pred_mask > THRESHOLD).astype(np.uint8)

    overlay = img_resized.copy()
    overlay[pred_mask_bin > 0] = [255, 0, 0]

    result = cv2.addWeighted(img_resized, 0.7, overlay, 0.3, 0)

    return result

# ================= UI =================
st.set_page_config(page_title="Bone Tumor Detection", layout="wide")

st.title("🦴 Bone Tumor Detection & Segmentation")
st.write("DenseNet + U-Net")

uploaded_file = st.file_uploader("Upload X-ray", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original Image", use_container_width=True)

    with col2:
        with st.spinner("Analyzing..."):
            prob = classify_image(image)

            if prob < 0.5:
                st.success("✅ No Tumor Detected")
                st.metric("Confidence", f"{1 - prob:.4f}")
            else:
                st.error("⚠️ Tumor Detected")
                st.metric("Confidence", f"{prob:.4f}")

                result = segment_image(image)
                st.image(result, caption="Tumor Segmentation", use_container_width=True)
