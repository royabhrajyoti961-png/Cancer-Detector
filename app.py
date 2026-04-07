import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import plotly.graph_objects as go
import time
import datetime

# --- 1. CORE SYSTEM CONFIGURATION ---
st.set_page_config(
    page_title="DERMA-LOGIC OS v4.5",
    page_icon="✚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. LIGHT DYNAMIC MEDICAL UI ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=JetBrains+Mono:wght@300;500&display=swap');

    /* Animated Light Gradient Background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(-45deg, #e6f7ff, #f0fbff, #ffffff, #e0f7fa);
        background-size: 400% 400%;
        animation: gradientFlow 12s ease infinite;
        font-family: 'JetBrains Mono', monospace;
        color: #003344;
    }

    @keyframes gradientFlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Glass Card (Light Mode) */
    .med-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(0, 150, 200, 0.2);
        border-radius: 14px;
        padding: 25px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
        margin-bottom: 20px;
    }

    /* Header */
    .clinical-header {
        font-family: 'Orbitron', sans-serif;
        color: #0077b6;
        text-transform: uppercase;
        letter-spacing: 2px;
        border-left: 5px solid #00b4d8;
        padding-left: 12px;
        margin-bottom: 20px;
    }

    /* Scan Box */
    .scan-box {
        position: relative;
        border: 2px solid #00b4d8;
        border-radius: 10px;
        overflow: hidden;
    }

    .scan-line {
        position: absolute;
        width: 100%;
        height: 3px;
        background: #00b4d8;
        box-shadow: 0 0 10px #00b4d8;
        animation: laser 2s infinite linear;
        z-index: 10;
    }

    @keyframes laser {
        0% { top: 0%; opacity: 0; }
        50% { opacity: 1; }
        100% { top: 100%; opacity: 0; }
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-thumb { background: #00b4d8; border-radius: 10px; }

    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- 3. MODEL ---
@st.cache_resource
def init_neural_engine():
    try:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(torch.load('skin_cancer_model.pth', map_location='cpu'))
        model.eval()
        return model
    except:
        return None

def preprocess_tensor(img):
    pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return pipeline(img).unsqueeze(0)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 style='color:#0077b6;'>✚ CONTROL</h2>", unsafe_allow_html=True)
    st.divider()

    scan_mode = st.radio("SENSITIVITY MODE", ["Standard", "High Precision", "Clinical Research"])

    st.markdown("---")
    st.subheader("SYSTEM VITALS")
    st.write(f"GPU: {'READY' if torch.cuda.is_available() else 'EMULATED'}")
    st.write(f"LATENCY: 42ms")
    st.progress(85, text="NEURAL LOAD")

    if st.button("REBOOT SYSTEM"):
        st.rerun()

# --- MAIN ---
engine = init_neural_engine()

c1, c2, c3 = st.columns([3, 1, 1])
with c1:
    st.markdown("<h1 class='clinical-header'>DERMA-LOGIC // DIAGNOSTIC TERMINAL</h1>", unsafe_allow_html=True)
with c2:
    st.metric("PULSE", "72 BPM", delta="Stable")
with c3:
    st.write(f"**DR. ABHRAJYOTI**  \n{datetime.datetime.now().strftime('%H:%M:%S')}")

st.divider()

left_panel, right_panel = st.columns([1, 1.4])

# --- LEFT PANEL ---
with left_panel:
    st.markdown('<div class="med-card">', unsafe_allow_html=True)

    upload = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])

    if upload:
        img_raw = Image.open(upload).convert('RGB')
        st.markdown('<div class="scan-box">', unsafe_allow_html=True)
        st.markdown('<div class="scan-line"></div>', unsafe_allow_html=True)
        st.image(img_raw, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Awaiting image input")

    st.markdown('</div>', unsafe_allow_html=True)

# --- RIGHT PANEL ---
with right_panel:
    st.markdown('<div class="med-card">', unsafe_allow_html=True)

    if upload and engine:
        with st.spinner("Analyzing..."):
            time.sleep(1)

            tensor = preprocess_tensor(img_raw)
            with torch.no_grad():
                logits = engine(tensor)
                probs = torch.nn.functional.softmax(logits, dim=1)
                risk = probs[0][1].item() * 100

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk,
                number={'suffix': "%"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#0077b6"},
                }
            ))

            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            if risk > 65:
                st.error("❗ High Risk Detected")
            elif risk > 35:
                st.warning("⚠️ Moderate Risk")
            else:
                st.success("✅ Low Risk")

            report_data = f"Date: {datetime.date.today()}\nRisk: {risk:.2f}%"
            st.download_button("Download Report", data=report_data, file_name="report.txt")

    else:
        st.info("Upload image to start analysis")

    st.markdown('</div>', unsafe_allow_html=True)

# --- FOOTER ---
st.caption("DERMA-LOGIC v4.5 | Light Medical Interface")
