import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

st.set_page_config(page_title="Performance - Fraud Dashboard", layout="wide")

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

render_nav("Performance")

st.markdown("<div class='page-title'>Model Performance Metrics</div>", unsafe_allow_html=True)

df = pd.read_csv("creditcard.csv")
model = joblib.load("fraud_model.pkl")

X = df.drop("Class", axis=1)
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)

accuracy = report["accuracy"]
precision = report["1"]["precision"]
recall = report["1"]["recall"]
f1 = report["1"]["f1-score"]

cols = st.columns(4)
values = [
    ("Accuracy", accuracy),
    ("Precision (Fraud)", precision),
    ("Recall (Fraud)", recall),
    ("F1 Score (Fraud)", f1),
]

for col, (label, value) in zip(cols, values):
    with col:
        st.markdown(
            f"<div class='glass-card'><div class='chart-title'>{label}</div>"
            f"<h2 style='color:#D4AF37;margin:0;'>{value:.2%}</h2></div>",
            unsafe_allow_html=True
        )

st.markdown("<div class='chart-title'>Confusion Matrix</div>", unsafe_allow_html=True)

fig, ax = plt.subplots()
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(ax=ax, colorbar=False)
st.pyplot(fig)
