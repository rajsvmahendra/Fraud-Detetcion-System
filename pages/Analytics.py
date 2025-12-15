import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="Analytics - Fraud Dashboard", layout="wide")

with open("styles/theme.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown(
    "<style>section[data-testid='stSidebar']{display:none;}</style>",
    unsafe_allow_html=True,
)

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
    st.markdown("</div>", unsafe_allow_html=True)

render_nav("Analytics")

st.markdown("<div class='page-title'>Analytics & Exploratory Data Analysis</div>", unsafe_allow_html=True)

df = pd.read_csv("creditcard.csv")

with st.expander("Raw Dataset (sample data set)"):
    st.dataframe(df.sample(300), use_container_width=True)

st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("<div class='chart-title'>Class Distribution (0 = Legit, 1 = Fraud)</div>", unsafe_allow_html=True)
fig1, ax1 = plt.subplots()
sns.countplot(x="Class", data=df, palette=["#00C985", "#D62839"], ax=ax1)
st.pyplot(fig1)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("<div class='chart-title'>Transaction Amount Distribution</div>", unsafe_allow_html=True)
fig2, ax2 = plt.subplots()
sns.histplot(df["Amount"], kde=True, ax=ax2, color="#D4AF37")
st.pyplot(fig2)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("<div class='chart-title'>Time vs Amount (sample 1000)</div>", unsafe_allow_html=True)
fig3, ax3 = plt.subplots()
sample_df = df.sample(min(1000, len(df)), random_state=42)
sns.scatterplot(
    x=sample_df["Time"],
    y=sample_df["Amount"],
    hue=sample_df["Class"],
    palette=["#00C985", "#D62839"],
    alpha=0.6,
    ax=ax3,
)
st.pyplot(fig3)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("<div class='chart-title'>Feature Correlation Heatmap</div>", unsafe_allow_html=True)
fig4, ax4 = plt.subplots(figsize=(14, 10))
sns.heatmap(df.drop("Class", axis=1).corr(), cmap="inferno", ax=ax4)
st.pyplot(fig4)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("<div class='chart-title'>Top Features vs Class (Violin Plots)</div>", unsafe_allow_html=True)
top = df.corr()["Class"].abs().sort_values(ascending=False)[1:6].index.tolist()
fig5, axs = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
axs = axs.flatten()
for i, feat in enumerate(top):
    sns.violinplot(data=df, x="Class", y=feat, palette=["#00C985", "#D62839"], ax=axs[i])
fig5.delaxes(axs[5])
st.pyplot(fig5)
st.markdown("</div>", unsafe_allow_html=True)
