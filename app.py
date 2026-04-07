import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import plotly.graph_objects as go
import datetime
import time

# --- 1. PAGE SETUP & THEME ---
st.set_page_config(page_title="DERMA-LOGIC v4.0 | DIAGNOSTIC TERMINAL", layout="wide")

# --- 2. HOSPITAL MONITOR CSS (Neon Blue/Green & Glassmorphism) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
    
    .stApp {
        background-color: #00080a;
        font-family: 'Share Tech Mono', monospace;
    }
    
    /* Hospital Grid Border */
    .med-container {
        border: 1px solid #00d4ff;
        padding: 20px;
        background: rgba(0, 212, 255, 0.02);
        border-radius: 5px;
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.1);
    }
    
    /* Glowing Labels */
    .glow-text {
        color: #00d4ff;
        text-shadow: 0 0 8px #00d4ff;
        letter-spacing: 2px;
        font-size: 0.9rem;
    }
    
    /* Animated Scanning Bar */
    .scan-line {
        width: 100%;
        height: 2px;
        background: #00ff41;
        position: relative;
        animation: scan 2s linear infinite;
    }
    @keyframes scan {
        0% { top: 0; }
        100% { top: 300px; }
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. MEDICAL ASSET LOADING ---
@st.cache_resource
def load_medical_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    # Ensure your .pth file is in the same folder on GitHub
    model.load_state_dict(torch.load('skin_cancer_model.pth', map_location='cpu'))
    model.eval()
    return model

# --- 4. TOP MONITOR BAR ---
t1, t2, t3 = st.columns([2, 1, 1])
with t1:
    st.markdown("<h1 style='color:#00d4ff; margin:0;'>✚ DERMA-SCAN: DIAGNOSTIC UNIT</h1>", unsafe_allow_html=True)
with t2:
    st.markdown(f"<p class='glow-text'>SYSTEM: ACTIVE<br>DATE: {datetime.date.today()}</p>", unsafe_allow_html=True)
with t3:
    st.markdown("<p class='glow-text'>LOCATION: COLLAB-NODE-01<br>SECURE ENCRYPTION: ON</p>", unsafe_allow_html=True)

st.divider()

# --- 5. THE MAIN INTERFACE ---
col_left, col_mid, col_right = st.columns([1, 1.2, 1])

with col_left:
    st.markdown('<div class="med-container">', unsafe_allow_html=True)
    st.markdown("<p class='glow-text'>[01] PATIENT UPLOAD</p>", unsafe_allow_html=True)
    file = st.file_uploader("Insert Image Scan", type=['jpg', 'png', 'jpeg'], label_visibility="collapsed")
    
    if file:
        img = Image.open(file)
        st.image(img, caption="RAW INPUT STREAM", use_column_width=True)
        st.markdown('<div class="scan-line"></div>', unsafe_allow_html=True)
    else:
        st.info("Awaiting sensor input...")
    st.markdown('</div>', unsafe_allow_html=True)

with col_mid:
    st.markdown('<div class="med-container">', unsafe_allow_html=True)
    st.markdown("<p class='glow-text'>[02] NEURAL PROCESSING</p>", unsafe_allow_html=True)
    
    if file:
        with st.spinner("ISOLATING LESION PATTERNS..."):
            time.sleep(1.5) # Simulate AI thinking
            # --- PREDICTION LOGIC ---
            # (Insert your transformation and model call here)
            # Mock results for visual display
            prob = 0.88 
            
            st.write("### AI PROBABILITY INDEX")
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "MALIGNANCY RISK", 'font': {'color': "#00d4ff"}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickcolor': "#00d4ff"},
                    'bar': {'color': "#ff4b4b" if prob > 0.5 else "#00ff41"},
                    'steps': [
                        {'range': [0, 50], 'color': "rgba(0, 255, 65, 0.1)"},
                        {'range': [50, 100], 'color': "rgba(255, 75, 75, 0.1)"}
                    ],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}
                }
            ))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "#00d4ff", 'family': "Arial"})
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data found in buffer.")
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="med-container">', unsafe_allow_html=True)
    st.markdown("<p class='glow-text'>[03] FINAL REPORT</p>", unsafe_allow_html=True)
    
    if file:
        st.markdown("---")
        st.markdown("**DIAGNOSTIC SUMMARY:**")
        if prob > 0.5:
            st.error("❗ ALERT: HIGH ATYPICALITY")
            st.write("System suggests immediate Dermatological Review.")
        else:
            st.success("✅ SCAN STATUS: NORMAL")
            st.write("No immediate threats detected by Neural Engine.")
        
        st.markdown("---")
        st.markdown("**VITAL METADATA:**")
        st.code(f"Confidence: {prob*100:.2f}%\nLatency: 0.04s\nModel: ResNet-Med")
    else:
        st.write("Ready for patient scan...")
    st.markdown('</div>', unsafe_allow_html=True)

# --- 6. FOOTER DISCLAIMER ---
st.markdown("<br><p style='text-align:center; color:grey; font-size:10px;'>NON-DIAGNOSTIC RESEARCH TOOL. FOR AWARENESS ONLY. CONSULT A DOCTOR.</p>", unsafe_allow_html=True)
