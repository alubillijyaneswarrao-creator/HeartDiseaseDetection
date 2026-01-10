import streamlit as st
import requests
import pandas as pd
import numpy as np
import joblib
import shap
import os

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from io import BytesIO


# ===============================
# LOAD SHAP FILES (DEPLOYMENT SAFE)
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

shap_values_global = joblib.load(
    os.path.join(ROOT_DIR, "shap_explainer", "shap_values.pkl")
)

shap_feature_names = joblib.load(
    os.path.join(ROOT_DIR, "shap_explainer", "shap_feature_names.pkl")
)


# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Heart Disease Dashboard",
    page_icon="❤️",
    layout="wide"
)

# ===============================
# 🌈 CUSTOM UI THEME & ANIMATIONS
# ===============================



# ===============================
# 🌗 THEME TOGGLE
# ===============================
theme = st.sidebar.radio(
    "🎨 Theme Mode",
    ["☀️ Light", "🌙 Dark"],
    horizontal=True
)

if theme == "☀️ Light":
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #cfe9ff, #e8f4ff);
        color: #0f172a;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #b6ddff, #dceeff);
    }

    h1,h2,h3,h4,h5,p,span,label {
        color: #0f172a;
    }

    input, textarea, select {
        background-color: #ffffff;
        color: #0f172a;
    }

    button, button[kind="primary"] {
        background: linear-gradient(135deg, #2563eb, #1e40af);
        color: white;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #020617, #0f172a);
        color: #f8fafc;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #020617, #020617);
    }

    section[data-testid="stSidebar"] * {
        color: #f8fafc;
    }

    h1,h2,h3,h4,h5,p,span,label {
        color: #f8fafc;
    }

    input, textarea, select {
        background-color: #020617;
        color: #f8fafc;
        border: 1px solid #334155;
    }

    button, button[kind="primary"] {
        background: linear-gradient(135deg, #7c3aed, #9333ea);
        color: white;
        border-radius: 10px;
    }
    button:hover {
    opacity: 0.9;
    transform: scale(1.02);
}
    </style>
    """, unsafe_allow_html=True)


# ===============================
# Load precomputed SHAP values
# ===============================
shap_values_global = joblib.load("shap_explainer/shap_values.pkl")
shap_feature_names = joblib.load("shap_explainer/shap_feature_names.pkl")

st.title("🫀 Heart Disease Prediction Dashboard")
st.caption("Interactive ML-based medical decision support system")

# ===============================
# SIDEBAR — PATIENT INPUT
# ===============================
save_record = st.sidebar.button("💾 Save Patient Record")
if save_record:
    st.toast("✅ Patient record saved successfully", icon="💾")

st.sidebar.header("🧍 Patient Details")

patient_name = st.sidebar.text_input("Patient Name")

age = st.sidebar.slider("Age", 20, 90, 55)
sex = st.sidebar.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
trestbps = st.sidebar.slider("Resting BP (mm Hg)", 80, 200, 130)
chol = st.sidebar.slider("Cholesterol (mg/dl)", 100, 600, 240)
fbs = st.sidebar.selectbox("Fasting Blood Sugar >120", [0, 1])
thalach = st.sidebar.slider("Max Heart Rate", 60, 220, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.sidebar.slider("ST Depression", 0.0, 6.0, 1.2, 0.1)
slope = st.sidebar.selectbox("Slope of ST Segment", [0, 1, 2])
ca = st.sidebar.selectbox("Major Vessels (0–4)", [0, 1, 2, 3, 4])
cp = st.sidebar.selectbox("Chest Pain Type", [0, 1, 2, 3])
restecg = st.sidebar.selectbox("Rest ECG", [0, 1, 2])
thal = st.sidebar.selectbox("Thalium Stress Test", [0, 1, 2, 3])

# ===============================
# API CALL (AUTO-UPDATE STYLE)
# ===============================
payload = {
    "age": age,
    "sex": sex,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs,
    "thalach": thalach,
    "exang": exang,
    "oldpeak": oldpeak,
    "slope": slope,
    "ca": ca,
    "cp": cp,
    "restecg": restecg,
    "thal": thal
}

prediction = None
probability = None
result = {}

with st.spinner("🔄 Analyzing patient vitals..."):
    try:
        response = requests.post("https://heartdiseasedetection-njhi.onrender.com/predict", json=payload)
        result = response.json()
        prediction = result.get("prediction")
        probability = result.get("probability")
    except:
        st.error("⚠️ Backend not running")
        st.stop()


risk_percent = probability * 100 if probability is not None else 0
import os
from datetime import datetime

# ===============================
# SAVE PATIENT HISTORY
# ===============================
if save_record and prediction is not None and probability is not None:
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "name": patient_name,
        "age": age,
        "sex": "Male" if sex == 0 else "Female",
        "bp": trestbps,
        "cholesterol": chol,
        "heart_rate": thalach,
        "risk_percent": round(probability * 100, 2),
        "prediction": "Heart Disease" if prediction == 1 else "No Disease"
    }

    history_file = "patient_history.csv"

    if os.path.exists(history_file):
        df_hist = pd.read_csv(history_file)
        df_hist = pd.concat(
            [df_hist, pd.DataFrame([record])],
            ignore_index=True
        )
    else:
        df_hist = pd.DataFrame([record])

    df_hist.to_csv(history_file, index=False)

if probability is None:
    st.stop()


def generate_pdf_report(patient_data):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    y = height - 50

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, y, "Heart Disease Medical Report")

    y -= 40
    c.setFont("Helvetica", 11)

    for key, value in patient_data.items():
        c.drawString(50, y, f"{key}: {value}")
        y -= 18

    y -= 20
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(
        50,
        y,
        "⚠️ This report is generated by an AI system and is for decision support only."
    )

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# ===============================
# MAIN DASHBOARD TABS
# ===============================
tab1, tab2, tab3, tab4 = st.tabs([
    "🧍 Patient Summary",
    "📊 Risk Analysis",
    "🧬 Patient vs Population",
    "📈 Model Confidence"
])

# ===============================
# TAB 1 — PATIENT SUMMARY
# ===============================
with tab1:
    st.subheader(f"🧍 Patient: {patient_name if patient_name else 'Unknown'}")
   

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Age", age)
        st.metric("Blood Pressure", f"{trestbps} mm Hg")
        st.metric("Cholesterol", f"{chol} mg/dl")

    with col2:
        if prediction is not None:
            if prediction == 1:
                st.error("⚠️ Heart Disease Detected")
            else:
                st.success("✅ No Heart Disease Detected")


        st.metric("Heart Disease Risk", f"{risk_percent:.2f}%")
    st.markdown("### 🫀 Live Heart ECG (Simulated)")
    st.caption("Real-time cardiac waveform visualization")

    t, ecg = generate_ecg(thalach)

    ecg_fig = go.Figure()
    ecg_fig.add_trace(go.Scatter(
        x=t, y=ecg,
        mode='lines',
        line=dict(color='red', width=2)
    ))
    ecg_fig.update_layout(height=300)

    st.plotly_chart(ecg_fig, use_container_width=True)
    st.markdown("---")
    st.subheader("📁 Patient Visit History")

    if os.path.exists("patient_history.csv"):
        history_df = pd.read_csv("patient_history.csv")
        st.dataframe(history_df.tail(10), use_container_width=True)
    else:
        st.info("No patient history recorded yet.")
    st.markdown("### 📄 Download Medical Report")

    if prediction is not None:
        pdf_data = {
            "Patient Name": patient_name or "Unknown",
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Age": age,
            "Sex": "Male" if sex == 0 else "Female",
            "Blood Pressure": f"{trestbps} mm Hg",
            "Cholesterol": f"{chol} mg/dl",
            "Heart Rate": thalach,
            "Risk Percentage": f"{risk_percent:.2f} %",
            "Prediction": "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
        }

        pdf_buffer = generate_pdf_report(pdf_data)

        st.download_button(
            label="⬇️ Download PDF Medical Report",
            data=pdf_buffer,
            file_name=f"heart_report_{patient_name or 'patient'}.pdf",
            mime="application/pdf"
        )
    st.markdown("</div>", unsafe_allow_html=True)
# ===============================
# TAB 2 — RISK ANALYSIS (PIE + BAR)
# ===============================
with tab2:
    st.subheader("Risk Breakdown")

    if probability is not None:
        pie_fig = px.pie(
            names=["No Disease", "Heart Disease"],
            values=[100 - risk_percent, risk_percent],
            color_discrete_sequence=["#6EC1E4", "#E74C3C"]
        )
        st.plotly_chart(pie_fig, use_container_width=True)

        bar_fig = go.Figure()

        bar_fig.add_bar(
            x=["Heart Disease Risk"],
            y=[risk_percent],
            marker_color="red",
            name="Risk %"
        )

        # 🔴 70% danger threshold
        bar_fig.add_shape(
            type="line",
            x0=-0.5, x1=0.5,
            y0=70, y1=70,
            line=dict(color="black", width=3, dash="dash")
        )

        bar_fig.add_annotation(
            x=0,
            y=72,
            text="⚠️ High Risk Threshold (70%)",
            showarrow=False,
            font=dict(size=12)
        )

        bar_fig.update_layout(
            yaxis_range=[0, 100],
            height=350
        )

        st.plotly_chart(bar_fig, use_container_width=True)

    else:
        st.warning("Prediction not available")


# ===============================
# TAB 3 — PATIENT vs POPULATION
# ===============================
with tab3:
    st.subheader("Patient vs Typical Population")

    population = pd.DataFrame({
        "Feature": ["Age", "BP", "Cholesterol", "Heart Rate"],
        "Patient": [age, trestbps, chol, thalach],
        "Population Avg": [54, 131, 246, 150]
    })

    fig = px.bar(
        population,
        x="Feature",
        y=["Patient", "Population Avg"],
        barmode="group"
    )
    st.plotly_chart(fig, use_container_width=True)
# ===============================
# TAB 4 — MODEL CONFIDENCE & EXPLAINABILITY
# ===============================
with tab4:
    st.subheader("📈 Model Confidence & Explainability")

    # -------------------------------
    # Confidence Gauge
    # -------------------------------
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_percent,
        title={"text": "Heart Disease Risk (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "darkred"},
            "steps": [
                {"range": [0, 40], "color": "lightgreen"},
                {"range": [40, 70], "color": "orange"},
                {"range": [70, 100], "color": "red"}
            ]
        }
    ))
    st.plotly_chart(gauge, use_container_width=True)

    st.markdown("---")

    # -------------------------------
    # GLOBAL SHAP (Model-level)
    # -------------------------------
    st.subheader("🔍 Global Feature Importance (SHAP)")
    shap_mean = np.abs(shap_values_global.values).mean(axis=0)

    shap_exp_global = shap.Explanation(
        values=shap_mean,
        feature_names=shap_feature_names
    )


    fig1, ax1 = plt.subplots(figsize=(8, 4))
    shap.plots.bar(shap_exp_global, show=False)
    st.pyplot(fig1)

    st.markdown("---")

    # -------------------------------
    # LOCAL SHAP (Patient-level)
    # -------------------------------
    if "shap_values" in result:
        st.subheader("🧬 Why this patient got this prediction")

        shap_local = shap.Explanation(
            values=np.array(result["shap_values"]),
            feature_names=result["feature_names"]
        )

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        shap.plots.waterfall(shap_local, show=False)
        st.pyplot(fig2)
st.caption("⚠️ This tool is for decision support only and not a medical diagnosis.")



