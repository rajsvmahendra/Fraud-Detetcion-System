# ✅ Final Streamlit App: Fraud Detection Dashboard by Rajsv 💸

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from streamlit_lottie import st_lottie
import requests

# ------------------- Load Lottie Animation -------------------
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_json = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_t24tpvcu.json")

# ------------------- Page Config -------------------
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# ------------------- Custom CSS Styling -------------------
st.markdown("""
<style>
    .stApp {
        background-image: url('https://images.unsplash.com/photo-1605902711622-cfb43c4437d1');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .hero-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 70vh;
        text-align: center;
        color: white;
    }
    .hero-title {
        font-size: 3.5em;
        font-weight: bold;
        margin-bottom: 0.2em;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.6);
    }
    .hero-text {
        font-size: 1.4em;
        max-width: 800px;
        margin: 0 auto;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
</style>
""", unsafe_allow_html=True)

# ------------------- Hero Section -------------------
st.markdown("""
    <div class="hero-container">
        <div class="hero-title">&#128179; Credit Card Fraud Detection</div>
        <div class="hero-text">
            Welcome! This app simulates, monitors, and detects fraudulent financial transactions using machine learning.
        </div>
    </div>
""", unsafe_allow_html=True)

# Show Lottie animation (if loaded)
if lottie_json:
    st_lottie(lottie_json, height=200, key="intro-animation")
else:
    st.warning("⚠️ Animation failed to load. Check the Lottie URL or your internet connection.")

# Start Simulation
model = joblib.load("fraud_model.pkl")
df_real = pd.read_csv("creditcard.csv")

X = df_real.drop("Class", axis=1)
y = df_real["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

st.header("🔍 Manual Fraud Prediction")
with st.expander("Input 30 Features to Predict Fraud"):
    cols = st.columns(3)
    input_data = []
    for i in range(30):
        with cols[i % 3]:
            val = st.number_input(f"Feature {i+1}", format="%.5f", key=f"feature_{i+1}")
            input_data.append(val)

    if st.button("Predict Fraud"):
        columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        features_df = pd.DataFrame([input_data], columns=columns)
        prediction = model.predict(features_df)[0]
        if prediction == 1:
            st.error("🚨 Fraudulent Transaction Detected!")
        else:
            st.success("✅ Legitimate Transaction")

# ------------------- Simulated Transaction Generator -------------------
def simulate_from_real_data(df, n=100, fraud_ratio=0.15):
    fraud_df = df[df['Class'] == 1]
    legit_df = df[df['Class'] == 0]

    n_fraud = int(n * fraud_ratio)
    n_legit = n - n_fraud

    fraud_samples = fraud_df.sample(n=n_fraud)
    legit_samples = legit_df.sample(n=n_legit)

    combined_df = pd.concat([fraud_samples, legit_samples]).sample(frac=1).reset_index(drop=True)

    data = []
    for _, row in combined_df.iterrows():
        is_fraud = int(row['Class'])
        data.append({
            "customerName": f"User{random.randint(1000, 9999)}",
            "merchantName": f"Shop{random.randint(100, 999)}",
            "amount": round(row['Amount'], 2),
            "fraud": is_fraud,
            "fraudLevel": "high" if is_fraud == 1 else "low",
            "timestamp": pd.Timestamp.now() - pd.to_timedelta(random.randint(0, 3600), unit="s")
        })

    return pd.DataFrame(data)

df = simulate_from_real_data(df_real, 100)

# ------------------- Fraud Overview Metrics -------------------
st.header("📊 Fraud Overview")
fraud_count = df["fraud"].sum()
legit_count = len(df) - fraud_count
fraud_pct = round(fraud_count / len(df) * 100, 2)

col1, col2, col3 = st.columns(3)
col1.metric("Fraudulent Transactions", fraud_count, f"{fraud_pct}%")
col2.metric("Legitimate Transactions", legit_count)
col3.metric("Total Volume ($)", f"${df['amount'].sum():,.2f}")

# ------------------- Transaction Analytics -------------------
st.subheader("📈 Transaction Analytics")
c1, c2 = st.columns(2)
with c1:
    pie = px.pie(df, names="fraudLevel", title="Fraud Distribution")
    st.plotly_chart(pie, use_container_width=True)
with c2:
    df_sorted = df.sort_values("timestamp")
    line = px.line(df_sorted, x="timestamp", y="amount", title="Transaction Volume Over Time")
    st.plotly_chart(line, use_container_width=True)

c3, c4 = st.columns(2)
with c3:
    risky_merchants = df[df["fraud"] == 1]["merchantName"].value_counts().reset_index()
    risky_merchants.columns = ["merchant", "fraud_count"]
    if not risky_merchants.empty:
        bar = px.bar(risky_merchants, x="merchant", y="fraud_count", title="Top High-Risk Merchants")
        st.plotly_chart(bar, use_container_width=True)
    else:
        st.info("No high-risk merchants detected yet.")
with c4:
    st.subheader("🧾 Recent Transactions")
    st.dataframe(df[["customerName", "merchantName", "amount", "fraudLevel", "timestamp"]].head(10), use_container_width=True)

# ------------------- Deep EDA -------------------
st.header("🔬 Deep Data Visualization & EDA")
eda_df = df_real.copy()

with st.expander("🔍 Explore the Raw Data"):
    st.dataframe(eda_df.sample(100), use_container_width=True)

# ------------------- Class Distribution -------------------
st.subheader("📊 Class Distribution")
fig1, ax1 = plt.subplots(figsize=(10, 6))  # Adjusted size for better visibility
sns.countplot(data=df_real, x='Class', hue='Class', palette='coolwarm', edgecolor='black', ax=ax1)
ax1.set_title("Class Distribution", fontsize=18)
ax1.set_xlabel("Class (0 = Legit, 1 = Fraud)", fontsize=14)
ax1.set_ylabel("Count", fontsize=14)
ax1.tick_params(axis='both', labelsize=12)
st.pyplot(fig1)

# ------------------- Transaction Amount Distribution -------------------
st.subheader("💰 Transaction Amount Distribution")
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.histplot(df_real["Amount"], bins=50, kde=True, color='purple', ax=ax2)
ax2.set_title("Transaction Amount Distribution", fontsize=18)
ax2.set_xlabel("Transaction Amount ($)", fontsize=14)
ax2.set_ylabel("Frequency", fontsize=14)
ax2.tick_params(axis='both', labelsize=12)
st.pyplot(fig2)

# ------------------- Time vs Amount -------------------
st.subheader("⏰ Time vs Amount")
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.scatterplot(x="Time", y="Amount", hue="Class", data=df_real.sample(1000), palette="coolwarm", alpha=0.6, ax=ax3)
ax3.set_title("Time vs Amount (Transaction Scatter Plot)", fontsize=18)
ax3.set_xlabel("Time", fontsize=14)
ax3.set_ylabel("Transaction Amount ($)", fontsize=14)
ax3.tick_params(axis='both', labelsize=12)
st.pyplot(fig3)

# ------------------- Feature Correlation Heatmap -------------------
st.subheader("Feature Correlation Heatmap")

# Calculate the correlation matrix (excluding 'Class' for better clarity)
corr = df_real.drop("Class", axis=1).corr()

# Create the figure with a larger size
fig, ax = plt.subplots(figsize=(15, 12))

# Create the heatmap with annotations for correlation values
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax, cbar_kws={'label': 'Correlation Coefficient'})  

# Title and tweaks for better visibility
ax.set_title("Feature Correlation Heatmap", fontsize=18)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=14)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right', fontsize=14)

# Display the plot
st.pyplot(fig)

st.subheader("Violin Plots for Top Correlated Features")

top_corr_features = df_real.corr()["Class"].abs().sort_values(ascending=False)[1:6].index.tolist()

# Create a 2-row, 3-column grid and turn on constrained layout to avoid overlaps
fig, axs = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
axs = axs.flatten()

# Plot only the top 5 features
for i, feat in enumerate(top_corr_features):
    sns.violinplot(x="Class", y=feat, data=df_real, ax=axs[i], palette="Set2")
    axs[i].set_title(f"Feature: {feat}", fontsize=14)
    axs[i].set_xlabel("Class", fontsize=12)
    axs[i].set_ylabel(feat, fontsize=12)

# Hide the unused 6th plot
fig.delaxes(axs[5])

# Display the plot
st.pyplot(fig)


# Confusion Matrix
st.subheader("Confusion Matrix")
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
fig6, ax6 = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax6)
ax6.set_title("Confusion Matrix")
st.pyplot(fig6)
st.markdown("---")
st.caption("Built with ❤️ using Streamlit · Rajsv Mahendra 2025")
