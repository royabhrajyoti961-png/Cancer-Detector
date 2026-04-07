import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import datetime
import time
import sqlite3

# --- DATABASE ---
conn = sqlite3.connect("patients.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    age INTEGER,
    gender TEXT,
    date TEXT,
    risk REAL
)
""")
conn.commit()

# --- CONFIG ---
st.set_page_config(page_title="DermaLogic Hospital AI", page_icon="🏥", layout="wide")

# --- UI ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

html, body, [class*="css"], * {
    font-family: 'Poppins', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background-color: #f6f9fc;
}

.med-card {
    background: white;
    border-radius: 14px;
    padding: 20px;
    border: 1px solid #e6edf5;
    box-shadow: 0 6px 18px rgba(0,0,0,0.06);
    margin-bottom: 20px;
}

.header {
    font-size: 28px;
    font-weight: 600;
    color: #1b4965;
}

.section-title {
    font-size: 16px;
    font-weight: 600;
    color: #2a6f97;
}

.low {color: green;}
.mid {color: orange;}
.high {color: red;}
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

def preprocess(img):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return transform(img).unsqueeze(0)

# --- HEADER ---
st.markdown("<div class='header'>🏥 DermaLogic Hospital AI System</div>", unsafe_allow_html=True)
st.divider()

# --- PATIENT FORM ---
st.markdown("<div class='section-title'>Patient Details</div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
name = col1.text_input("Name")
age = col2.number_input("Age", min_value=1, max_value=120)
gender = col3.selectbox("Gender", ["Male", "Female", "Other"])

st.divider()

# --- LAYOUT ---
left, right = st.columns([1,1.2])

# --- LEFT ---
with left:
    st.markdown("<div class='med-card'>", unsafe_allow_html=True)

    upload = st.file_uploader("Upload Image", type=['jpg','png','jpeg'])

    if upload:
        image = Image.open(upload).convert("RGB")
        st.image(image, use_column_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --- RIGHT ---
with right:
    st.markdown("<div class='med-card'>", unsafe_allow_html=True)

    if upload and model:
        if st.button("Run Diagnosis"):
            with st.spinner("Analyzing..."):
                time.sleep(1)

                tensor = preprocess(image)
                with torch.no_grad():
                    out = model(tensor)
                    prob = torch.nn.functional.softmax(out, dim=1)
                    risk = prob[0][1].item() * 100

            st.metric("Risk Score", f"{risk:.2f}%")

            if risk > 65:
                st.markdown("<p class='high'>High Risk</p>", unsafe_allow_html=True)
            elif risk > 35:
                st.markdown("<p class='mid'>Moderate Risk</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p class='low'>Low Risk</p>", unsafe_allow_html=True)

            # SAVE DATA
            c.execute("INSERT INTO records (name, age, gender, date, risk) VALUES (?, ?, ?, ?, ?)",
                      (name, age, gender, str(datetime.date.today()), risk))
            conn.commit()

            st.success("Saved to database")

    else:
        st.info("Upload image and fill patient details")

    st.markdown("</div>", unsafe_allow_html=True)

# --- HISTORY ---
st.markdown("## 📊 Patient History")

search = st.text_input("Search by name")

if search:
    data = c.execute("SELECT * FROM records WHERE name LIKE ?", ('%' + search + '%',)).fetchall()
else:
    data = c.execute("SELECT * FROM records").fetchall()

st.dataframe(data)

# --- FOOTER ---
st.caption("Hospital AI System • Demo Project")
