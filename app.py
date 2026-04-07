import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import plotly.graph_objects as go
import time
import datetime

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="DermaLogic Clinical AI",
    page_icon="🏥",
    layout="wide"
)

# --- CLEAN MEDICAL UI ---
st.markdown("""
<style>
body {
    font-family: 'Segoe UI', sans-serif;
}

/* Background */
[data-testid="stAppViewContainer"] {
    background-color: #f4f8fb;
}

/* Card */
.med-card {
    background: white;
    border-radius: 12px;
    padding: 20px;
    border: 1px solid #e3eaf2;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}

/* Header */
.header {
    font-size: 26px;
    font-weight: 600;
    color: #1f4e79;
}

/* Section title */
.section-title {
    font-size: 16px;
    font-weight: 600;
    color: #2a6f97;
    margin-bottom: 10px;
}

/* Upload box */
.upload-box {
    border: 2px dashed #bcd4e6;
    padding: 40px;
    text-align: center;
    border-radius: 10px;
    color: #6c8aa6;
}

/* Status colors */
.low {color: #2e7d32; font-weight: 600;}
.mid {color: #f9a825; font-weight: 600;}
.high {color: #c62828; font-weight: 600;}
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

def preprocess(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return transform(img).unsqueeze(0)

model = load_model()

# --- HEADER ---
col1, col2 = st.columns([4,1])
with col1:
    st.markdown("<div class='header'>🏥 DermaLogic AI – Skin Cancer Screening System</div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"**{datetime.date.today()}**")

st.divider()

# --- LAYOUT ---
left, right = st.columns([1,1.2])

# --- LEFT PANEL ---
with left:
    st.markdown("<div class='med-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Patient Scan Upload</div>", unsafe_allow_html=True)

    upload = st.file_uploader("Upload skin image", type=['png','jpg','jpeg'])

    if upload:
        image = Image.open(upload).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
    else:
        st.markdown("<div class='upload-box'>Drag & Drop or Upload Image</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --- RIGHT PANEL ---
with right:
    st.markdown("<div class='med-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>AI Diagnostic Result</div>", unsafe_allow_html=True)

    if upload and model:
        with st.spinner("Running AI Analysis..."):
            time.sleep(1)

            tensor = preprocess(image)
            with torch.no_grad():
                output = model(tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                risk = probs[0][1].item() * 100

        # Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk,
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [0,100]},
                'bar': {'color': "#1f77b4"},
            }
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Result
        if risk > 65:
            st.markdown("<p class='high'>High Risk – Immediate medical consultation recommended</p>", unsafe_allow_html=True)
        elif risk > 35:
            st.markdown("<p class='mid'>Moderate Risk – Monitor and consult dermatologist</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='low'>Low Risk – No immediate concern</p>", unsafe_allow_html=True)

        # Report
        report = f"""
        DermaLogic Clinical Report
        Date: {datetime.date.today()}
        Risk Score: {risk:.2f}%
        """
        st.download_button("Download Report", report, "clinical_report.txt")

    else:
        st.info("Upload image to start diagnosis")

    st.markdown("</div>", unsafe_allow_html=True)

# --- FOOTER ---
st.caption("Clinical AI Tool | For Screening Purposes Only")
