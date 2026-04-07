import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import datetime
import time

# SAFE CV2 IMPORT
try:
    import cv2
    CV2_AVAILABLE = True
except:
    CV2_AVAILABLE = False

# --- CONFIG ---
st.set_page_config(page_title="DermaLogic AI", page_icon="🏥", layout="wide")

# --- UI (POPPINS + CLEAN HOSPITAL STYLE) ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap');

/* FORCE FONT */
html, body, [class*="css"], * {
    font-family: 'Poppins', sans-serif !important;
}

/* Background */
[data-testid="stAppViewContainer"] {
    background-color: #f6f9fc;
}

/* Card */
.med-card {
    background: white;
    border-radius: 14px;
    padding: 20px;
    border: 1px solid #e6edf5;
    box-shadow: 0 6px 18px rgba(0,0,0,0.06);
    margin-bottom: 20px;
}

/* Header */
.header {
    font-size: 28px;
    font-weight: 600;
    color: #1b4965;
}

/* Section */
.section-title {
    font-size: 16px;
    font-weight: 600;
    color: #2a6f97;
    margin-bottom: 10px;
}

/* File uploader fix */
[data-testid="stFileUploader"] {
    border: 2px dashed #c5d9e8;
    padding: 20px;
    border-radius: 10px;
    background-color: #f9fcff;
}

[data-testid="stFileUploader"] button {
    background-color: #1b4965;
    color: white;
    border-radius: 8px;
}

[data-testid="stFileUploader"] label {
    display: none;
}

/* Status */
.low {color: #2e7d32; font-weight: 600;}
.mid {color: #f9a825; font-weight: 600;}
.high {color: #d32f2f; font-weight: 600;}
</style>
""", unsafe_allow_html=True)

# --- MODEL ---
@st.cache_resource
def load_model():
    try:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(torch.load('skin_cancer_model.pth', map_location='cpu'))
        model.eval()
        return model
    except:
        return None

model = load_model()

# --- PREPROCESS ---
def preprocess(img):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return transform(img).unsqueeze(0)

# --- HEATMAP ---
def generate_heatmap(model, image):
    if not CV2_AVAILABLE:
        return None

    model.eval()
    img_tensor = preprocess(image)
    img_tensor.requires_grad = True

    features = []
    gradients = []

    def forward_hook(module, input, output):
        features.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    layer = model.layer4
    handle_f = layer.register_forward_hook(forward_hook)
    handle_b = layer.register_backward_hook(backward_hook)

    output = model(img_tensor)
    class_idx = output.argmax()

    model.zero_grad()
    output[0, class_idx].backward()

    grads = gradients[0].detach().numpy()[0]
    fmap = features[0].detach().numpy()[0]

    weights = np.mean(grads, axis=(1,2))
    cam = np.zeros(fmap.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * fmap[i]

    cam = np.maximum(cam, 0)
    cam = cam / cam.max()

    import cv2
    cam = cv2.resize(cam, (224,224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    original = cv2.resize(np.array(image), (224,224))

    return cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

# --- HEADER ---
col1, col2 = st.columns([4,1])
with col1:
    st.markdown("<div class='header'>🏥 DermaLogic AI – Clinical Skin Screening</div>", unsafe_allow_html=True)
with col2:
    st.write(f"**{datetime.date.today()}**")

st.divider()

# --- LAYOUT ---
left, right = st.columns([1,1.2])

# --- LEFT ---
with left:
    st.markdown("<div class='med-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Upload Patient Image</div>", unsafe_allow_html=True)

    upload = st.file_uploader(
        "Upload Patient Image",
        type=['jpg','png','jpeg'],
        label_visibility="collapsed"
    )

    if upload:
        image = Image.open(upload).convert("RGB")
        st.image(image, use_column_width=True)

    else:
        st.info("Drag & drop or upload image")

    st.markdown("</div>", unsafe_allow_html=True)

# --- RIGHT ---
with right:
    st.markdown("<div class='med-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>AI Diagnosis</div>", unsafe_allow_html=True)

    if upload and model:
        with st.spinner("Analyzing..."):
            time.sleep(1)

            tensor = preprocess(image)
            with torch.no_grad():
                out = model(tensor)
                prob = torch.nn.functional.softmax(out, dim=1)
                risk = prob[0][1].item() * 100

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk,
            number={'suffix': "%"},
            gauge={'axis': {'range': [0,100]}}
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        if risk > 65:
            st.markdown("<p class='high'>High Risk</p>", unsafe_allow_html=True)
        elif risk > 35:
            st.markdown("<p class='mid'>Moderate Risk</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='low'>Low Risk</p>", unsafe_allow_html=True)

        st.markdown("### 🔬 AI Heatmap Analysis")
        heatmap_img = generate_heatmap(model, image)

        if heatmap_img is not None:
            st.image(heatmap_img)
        else:
            st.warning("Heatmap unavailable (cv2 not installed)")

        report = f"Date: {datetime.date.today()}\nRisk: {risk:.2f}%"
        st.download_button("Download Report", report, "report.txt")

    else:
        st.info("Upload image to begin")

    st.markdown("</div>", unsafe_allow_html=True)

# --- FOOTER ---
st.caption("AI Clinical Tool • Not a substitute for professional diagnosis")
