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

gdown.download("https://drive.google.com/uc?id=1gGNsPyeDb-oLQ0K14HFNYpWW4auohWzg", DENSENET_PATH, quiet=True, fuzzy=True)
gdown.download("https://drive.google.com/uc?id=1cKucZBoFr5sL6VQoc3YawSri69nvwuPp", UNET_PATH, quiet=True, fuzzy=True)

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

# ================= PREPROCESS (SAME AS HIS) =================
def preprocess_image(image):
    img_array = np.array(image)

    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

    img_pil = Image.fromarray(img_array)
    img_rgb = img_pil.convert("RGB").resize((IMG_SIZE, IMG_SIZE))

    img_arr = np.array(img_rgb) / 255.0
    input_arr = np.expand_dims(img_arr, axis=0)

    return input_arr, img_rgb

# ================= CLASSIFICATION =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def classify(image):
    img = image.convert("RGB").resize((224, 224))
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        out = model_dense(img)
        prob = torch.softmax(out, dim=1)[0][1].item()

    return prob

# ================= SEGMENTATION (HIS EXACT STYLE) =================
def segment(image):
    input_arr, img_rgb = preprocess_image(image)

    pred_mask = model_unet.predict(input_arr)[0].squeeze()
    pred_mask_bin = (pred_mask > THRESHOLD).astype(np.uint8)

    # contours
    pred_mask_uint8 = (pred_mask_bin * 255).astype(np.uint8)
    contours, _ = cv2.findContours(pred_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # filter small noise (HIS logic)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 10]

    img_viz = np.array(img_rgb).copy()

    for cnt in contours:
        # red contour
        cv2.drawContours(img_viz, [cnt], -1, (255, 0, 0), 2)

        # green box
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img_viz, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return img_viz, pred_mask, len(contours)

# ================= UI =================
st.title("🦴 Bone Tumor Detection (Fixed)")

file = st.file_uploader("Upload X-ray", type=["png","jpg","jpeg"])

if file:
    image = Image.open(file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original")

    with col2:
        with st.spinner("Analyzing..."):
            prob = classify(image)

            if prob < 0.5:
                st.success("✅ Normal")
                st.write(f"Confidence: {1-prob:.3f}")

            else:
                st.error("⚠️ Tumor Detected")
                st.write(f"Confidence: {prob:.3f}")

                result, heatmap, count = segment(image)

                st.image(result, caption=f"Detected Regions: {count}")
                st.image(heatmap, caption="Heatmap")
