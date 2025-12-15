import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Prediction - Fraud Dashboard", layout="wide")

with open("styles/theme.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("<style>section[data-testid='stSidebar']{display:none;}</style>", unsafe_allow_html=True)

def render_nav(active: str):
    st.markdown("<div class='nav-row'>", unsafe_allow_html=True)
    cols = st.columns(5)
    labels = ["Home", "Overview", "Prediction", "Analytics", "Performance"]
    targets = [
        "app.py",
        "pages/Overview.py",
        "pages/Predict.py",
        "pages/Analytics.py",
        "pages/Model_Performance.py",
    ]

    for col, label, target in zip(cols, labels, targets):
        with col:
            if label == active:
                st.markdown(f"<div class='nav-pill-active'>{label}</div>", unsafe_allow_html=True)
            else:
                if st.button(label, key=f"nav_{label}"):
                    st.switch_page(target if label != "Home" else "app.py")
render_nav("Prediction")

st.markdown("<div class='page-title'>Predict Transaction Fraud</div>", unsafe_allow_html=True)

model = joblib.load("fraud_model.pkl")

features = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
inputs = []

col = st.columns(3)
for idx, name in enumerate(features):
    with col[idx % 3]:
        value = st.number_input(name, value=0.0, format="%.5f")
        inputs.append(value)

if st.button("Run Prediction"):
    df = pd.DataFrame([inputs], columns=features)
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0][1]

    if pred == 1:
        st.error(f"Fraud Detected — Risk Score: {proba:.1%}")
    else:
        st.success(f"Legitimate Transaction — Fraud Probability: {proba:.1%}")
