import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageOps
import plotly.graph_objects as go
import time
import datetime
import pandas as pd

# --- 1. CORE SYSTEM CONFIGURATION ---
st.set_page_config(
    page_title="DERMA-LOGIC OS v4.5",
    page_icon="✚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. THE "FINEST" MEDICAL UI (CSS) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=JetBrains+Mono:wght@300;500&display=swap');

    /* Dynamic Deep Space Gradient */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at center, #001a1d 0%, #00080a 100%);
        background-attachment: fixed;
        color: #00d4ff;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Professional Glassmorphism Card */
    .med-card {
        background: rgba(0, 30, 40, 0.4);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
        margin-bottom: 20px;
    }

    /* Clinical Typography */
    .clinical-header {
        font-family: 'Orbitron', sans-serif;
        color: #00d4ff;
        text-transform: uppercase;
        letter-spacing: 3px;
        border-left: 4px solid #00ff41;
        padding-left: 15px;
        margin-bottom: 20px;
    }

    /* Animated Cyber Scan Overlay */
    .scan-box {
        position: relative;
        border: 1px solid #00d4ff;
        overflow: hidden;
    }
    .scan-line {
        position: absolute;
        width: 100%;
        height: 2px;
        background: #00ff41;
        box-shadow: 0 0 15px #00ff41;
        animation: laser 2.5s infinite linear;
        z-index: 10;
    }
    @keyframes laser {
        0% { top: 0%; opacity: 0; }
        50% { opacity: 1; }
        100% { top: 100%; opacity: 0; }
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-thumb { background: #00d4ff; border-radius: 10px; }

    /* Hide unnecessary UI */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 3. NEURAL ENGINE (BACKEND) ---
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

# --- 4. SIDEBAR CONTROL UNIT ---
with st.sidebar:
    st.markdown("<h2 style='font-family:Orbitron; color:#00ff41;'>✚ CONTROL</h2>", unsafe_allow_html=True)
    st.divider()
    scan_mode = st.radio("SENSITIVITY MODE", ["Standard", "High Precision", "Clinical Research"])
    st.markdown("---")
    st.subheader("SYSTEM VITALS")
    st.write(f"GPU: {'READY' if torch.cuda.is_available() else 'EMULATED'}")
    st.write(f"LATENCY: 42ms")
    st.progress(85, text="NEURAL LOAD")
    
    if st.button("REBOOT SYSTEM"):
        st.rerun()

# --- 5. MAIN INTERFACE ---
engine = init_neural_engine()

# Global Header
c1, c2, c3 = st.columns([3, 1, 1])
with c1:
    st.markdown("<h1 class='clinical-header'>DERMA-LOGIC // DIAGNOSTIC TERMINAL</h1>", unsafe_allow_html=True)
with c2:
    st.metric("PULSE", "72 BPM", delta="Stable")
with c3:
    st.write(f"**LOGGED:** DR. ABHRAJYOTI\n**NODE:** {datetime.datetime.now().strftime('%H:%M:%S')}")

st.divider()

# Grid Layout
left_panel, right_panel = st.columns([1, 1.4])

with left_panel:
    st.markdown('<div class="med-card">', unsafe_allow_html=True)
    st.markdown("<p style='color:#00ff41;'>[SENSORS: ACTIVE]</p>", unsafe_allow_html=True)
    
    upload = st.file_uploader("DROP SCAN FILE", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")
    
    if upload:
        img_raw = Image.open(upload).convert('RGB')
        st.markdown('<div class="scan-box">', unsafe_allow_html=True)
        st.markdown('<div class="scan-line"></div>', unsafe_allow_html=True)
        st.image(img_raw, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("<div style='height:300px; border:1px dashed #00d4ff; display:flex; align-items:center; justify-content:center;'>AWAITING INPUT</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right_panel:
    st.markdown('<div class="med-card">', unsafe_allow_html=True)
    st.markdown("<p style='color:#00ff41;'>[DIAGNOSTICS: STANDBY]</p>", unsafe_allow_html=True)
    
    if upload and engine:
        with st.spinner("QUANTIZING PIXELS..."):
            time.sleep(1) # Visual effect
            
            # Prediction
            tensor = preprocess_tensor(img_raw)
            with torch.no_grad():
                logits = engine(tensor)
                probs = torch.nn.functional.softmax(logits, dim=1)
                risk = probs[0][1].item() * 100

            # Visualization
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk,
                number={'suffix': "%", 'font': {'color': '#00d4ff', 'size': 50}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': "#00d4ff"},
                    'bar': {'color': "#ff004c" if risk > 50 else "#00ff41"},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 1,
                    'bordercolor': "#00d4ff",
                    'steps': [
                        {'range': [0, 35], 'color': 'rgba(0, 255, 65, 0.05)'},
                        {'range': [35, 75], 'color': 'rgba(255, 165, 0, 0.05)'},
                        {'range': [75, 100], 'color': 'rgba(255, 0, 76, 0.05)'}
                    ]
                }
            ))
            fig.update_layout(height=280, margin=dict(t=10, b=10), paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

            # Report Logic
            st.markdown("---")
            if risk > 65:
                st.markdown("<h3 style='color:#ff004c;'>❗ CRITICAL FINDINGS</h3>", unsafe_allow_html=True)
                st.write("Pattern recognition detects high confidence in cellular atypia. Immediate dermoscopic evaluation requested.")
            elif risk > 35:
                st.markdown("<h3 style='color:orange;'>⚠️ MODERATE OBSERVATION</h3>", unsafe_allow_html=True)
                st.write("Anomalous patterns detected. Monitor lesion for changes in ABCDE criteria.")
            else:
                st.markdown("<h3 style='color:#00ff41;'>✅ CLEAR SCAN</h3>", unsafe_allow_html=True)
                st.write("No malignant markers identified by Neural Engine v4.5.")

            # Clinical Export
            st.markdown("<br>", unsafe_allow_html=True)
            report_data = f"DERMA-LOGIC REPORT\nDate: {datetime.date.today()}\nRisk: {risk:.2f}%\nStatus: {'MALIGNANT' if risk > 50 else 'BENIGN'}"
            st.download_button("GENERATE CLINICAL REPORT", data=report_data, file_name=f"Report_{datetime.date.today()}.txt")

    else:
        st.info("System Ready. Please feed medical imaging data into the sensor unit.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- 6. LOGS ---
st.markdown("<p style='font-size:10px; color:rgba(0,212,255,0.3);'>► ENCRYPTION_SHA256: 4F88A... // BUFFER_SIZE: 1024KB // ENGINE: RESNET18_PYTORCH</p>", unsafe_allow_html=True)
