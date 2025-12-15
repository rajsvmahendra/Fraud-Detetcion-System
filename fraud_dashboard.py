import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import random
import base64
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Fraud Intelligence Dashboard", layout="wide")

# ------------- LOAD CSS ----------------
def load_css():
    with open("styles/theme.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ------------- BACKGROUND VIDEO --------------
def background_video(path):
    with open(path, "rb") as video_file:
        video_bytes = video_file.read()
    encoded_video = base64.b64encode(video_bytes).decode()
    st.markdown(
        f"""
        <video id="bg-video" autoplay loop muted playsinline>
            <source src="data:video/mp4;base64,{encoded_video}" type="video/mp4">
        </video>
        """,
        unsafe_allow_html=True
    )

background_video("assets/14250435_1920_1080_30fps.mp4")


# ---------- LOAD DATA & MODEL ----------
model = joblib.load("fraud_model.pkl")
df_real = pd.read_csv("creditcard.csv")

X = df_real.drop("Class", axis=1)
y = df_real["Class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


# -------------- SIMULATE TRANSACTIONS -------------
def simulate(df, count=100, fraud_ratio=0.15):
    fraud_df = df[df["Class"] == 1]
    legit_df = df[df["Class"] == 0]

    f = int(count * fraud_ratio)
    l = count - f

    combined = pd.concat([
        fraud_df.sample(f, random_state=42),
        legit_df.sample(l, random_state=42)
    ]).sample(frac=1, random_state=42)

    rows = []
    for _, r in combined.iterrows():
        rows.append({
            "Customer": f"CUST-{random.randint(1000,9999)}",
            "Merchant": f"SHOP-{random.randint(100,999)}",
            "Amount": round(r["Amount"], 2),
            "Class": int(r["Class"]),
            "RiskLevel": "High" if r["Class"] == 1 else "Low",
            "Time": pd.Timestamp.now() - pd.to_timedelta(random.randint(0, 6000), unit="s")
        })
    return pd.DataFrame(rows)

df = simulate(df_real)

# ----------------- TITLE -----------------
st.markdown("<h1 class='title'>Credit Card Fraud Intelligence Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Real-time monitoring, analytics and model evaluation</p>", unsafe_allow_html=True)


# ----------------- KPIs -----------------
fraud_count = df["Class"].sum()
legit_count = len(df) - fraud_count
total_volume = df["Amount"].sum()
avg_amount = df["Amount"].mean()

c1, c2, c3, c4 = st.columns(4)
c1.markdown(f"<div class='glass metric'><h3>{fraud_count}</h3><p>Fraud Cases</p></div>", unsafe_allow_html=True)
c2.markdown(f"<div class='glass metric'><h3>{legit_count}</h3><p>Legitimate</p></div>", unsafe_allow_html=True)
c3.markdown(f"<div class='glass metric'><h3>${total_volume:,.2f}</h3><p>Total Volume</p></div>", unsafe_allow_html=True)
c4.markdown(f"<div class='glass metric'><h3>${avg_amount:,.2f}</h3><p>Average Amount</p></div>", unsafe_allow_html=True)


# -------------------------------- OVERVIEW CHARTS -----------------------------
st.markdown("<h2 class='section'>Overview Analytics</h2>", unsafe_allow_html=True)

l1, l2 = st.columns(2)

with l1:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    pie = px.pie(
        df, names="RiskLevel",
        color="RiskLevel",
        color_discrete_map={"High": "#D62839", "Low": "#00C985"}
    )
    pie.update_layout(title="Fraud vs Legit Distribution")
    st.plotly_chart(pie, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with l2:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    tdf = df.sort_values("Time")
    line = px.line(
        tdf, x="Time", y="Amount", color="RiskLevel",
        color_discrete_map={"High": "#D62839", "Low": "#00C985"},
        title="Transaction Amount Over Time"
    )
    st.plotly_chart(line, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ------------------- Recent Transactions -------------------
st.markdown("<h2 class='section'>Recent Transactions</h2>", unsafe_allow_html=True)
st.markdown("<div class='glass'>", unsafe_allow_html=True)
st.dataframe(df.sort_values("Time", ascending=False).head(15), use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)


# ----------------- RAW DATA EXPLORER -----------------
st.markdown("<h2 class='section'>Explore Raw Credit Card Dataset</h2>", unsafe_allow_html=True)
with st.expander("Open Raw Dataset (sample 200 rows)"):
    st.dataframe(df_real.sample(200), use_container_width=True)


# ----------------- DEEP EDA -----------------
st.markdown("<h2 class='section'>Exploratory Data Analysis</h2>", unsafe_allow_html=True)

# class distribution
st.markdown("<p class='label'>Fraud vs Legit Class Balance</p>", unsafe_allow_html=True)
st.markdown("<div class='glass'>", unsafe_allow_html=True)
fig1, ax1 = plt.subplots()
sns.countplot(data=df_real, x="Class", palette=["#00C985", "#D62839"], ax=ax1)
ax1.set_xlabel("Class (0=Legit, 1=Fraud)")
ax1.set_ylabel("Count")
st.pyplot(fig1)
st.markdown("</div>", unsafe_allow_html=True)

# histogram
st.markdown("<p class='label'>Transaction Amount Distribution</p>", unsafe_allow_html=True)
st.markdown("<div class='glass'>", unsafe_allow_html=True)
fig2, ax2 = plt.subplots()
sns.histplot(df_real["Amount"], bins=50, kde=True, color="#D4AF37", ax=ax2)
ax2.set_xlabel("Amount ($)")
ax2.set_ylabel("Frequency")
st.pyplot(fig2)
st.markdown("</div>", unsafe_allow_html=True)

# scatter
st.markdown("<p class='label'>Time vs Amount Relationship</p>", unsafe_allow_html=True)
st.markdown("<div class='glass'>", unsafe_allow_html=True)
fig3, ax3 = plt.subplots()
sns.scatterplot(data=df_real.sample(1000), x="Time", y="Amount", hue="Class", ax=ax3, alpha=0.6,
                palette=["#00C985", "#D62839"])
st.pyplot(fig3)
st.markdown("</div>", unsafe_allow_html=True)

# heatmap
st.markdown("<p class='label'>Feature Correlation Heatmap</p>", unsafe_allow_html=True)
st.markdown("<div class='glass'>", unsafe_allow_html=True)
fig4, ax4 = plt.subplots(figsize=(14, 10))
sns.heatmap(df_real.drop("Class", axis=1).corr(), cmap="inferno", ax=ax4)
st.pyplot(fig4)
st.markdown("</div>", unsafe_allow_html=True)

# violin
st.markdown("<p class='label'>Top Features by Correlation to Fraud</p>", unsafe_allow_html=True)
st.markdown("<div class='glass'>", unsafe_allow_html=True)
top = df_real.corr()["Class"].abs().sort_values(ascending=False)[1:6].index.tolist()
fig5, axs = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
axs = axs.flatten()
for i, feature in enumerate(top):
    sns.violinplot(data=df_real, x="Class", y=feature, palette=["#00C985", "#D62839"], ax=axs[i])
fig5.delaxes(axs[5])
st.pyplot(fig5)
st.markdown("</div>", unsafe_allow_html=True)


# ------------------ MODEL PERFORMANCE ------------------
st.markdown("<h2 class='section'>Model Performance</h2>", unsafe_allow_html=True)
st.markdown("<div class='glass'>", unsafe_allow_html=True)

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
fig6, ax6 = plt.subplots()
disp = ConfusionMatrixDisplay(cm)
disp.plot(ax=ax6, cmap="cividis", colorbar=False)
ax6.set_title("Confusion Matrix")
st.pyplot(fig6)

st.markdown("</div>", unsafe_allow_html=True)


st.markdown("<p class='footer'>Dashboard by Rajsv Mahendra</p>", unsafe_allow_html=True)
