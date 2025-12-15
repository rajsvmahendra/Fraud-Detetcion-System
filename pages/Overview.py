import streamlit as st
import pandas as pd
import random
import plotly.express as px

st.set_page_config(page_title="Overview - Fraud Dashboard", layout="wide")

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

    st.markdown("</div>", unsafe_allow_html=True)

render_nav("Overview")

st.markdown("<div class='page-title'>Real-Time Fraud Monitoring</div>", unsafe_allow_html=True)

df_real = pd.read_csv("creditcard.csv")

def simulate(df, n=100, fraud_ratio=0.15):
    fraud = df[df["Class"] == 1]
    legit = df[df["Class"] == 0]
    f = int(n * fraud_ratio)
    l = n - f

    combined = pd.concat([fraud.sample(f), legit.sample(l)]).sample(frac=1)
    rows = []
    for _, row in combined.iterrows():
        rows.append({
            "Customer": f"CUST-{random.randint(1000,9999)}",
            "Merchant": f"SHOP-{random.randint(100,999)}",
            "Amount": round(row["Amount"], 2),
            "Fraud": int(row["Class"]),
            "Risk": "High" if int(row["Class"]) == 1 else "Low",
            "Time": pd.Timestamp.now() - pd.to_timedelta(random.randint(0, 3600), unit="s")
        })

    return pd.DataFrame(rows)

if "overview_df" not in st.session_state:
    st.session_state["overview_df"] = simulate(df_real)

if st.button("Refresh Live Feed"):
    st.session_state["overview_df"] = simulate(df_real)

df = st.session_state["overview_df"]

fraud = df["Fraud"].sum()
legit = len(df) - fraud
total = df["Amount"].sum()
avg = df["Amount"].mean()

c1, c2, c3, c4 = st.columns(4)
metrics = [
    ("Fraudulent", fraud),
    ("Legitimate", legit),
    ("Total Volume ($)", f"{total:,.2f}"),
    ("Avg Transaction ($)", f"{avg:,.2f}"),
]
for col, (label, value) in zip([c1, c2, c3, c4], metrics):
    with col:
        st.markdown(
            f"<div class='glass-card'><div class='chart-title'>{label}</div>"
            f"<h2 style='color:#D4AF37;margin:0;'>{value}</h2></div>",
            unsafe_allow_html=True
        )

st.markdown("<div class='chart-title'>Live Transactions Table</div>", unsafe_allow_html=True)
st.dataframe(df.sort_values("Time", ascending=False), use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    st.markdown("<div class='chart-title'>Fraud vs Legit Distribution</div>", unsafe_allow_html=True)
    fig = px.pie(df, names="Risk", color="Risk",
                 color_discrete_map={"High": "#D62839", "Low": "#00C985"})
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.markdown("<div class='chart-title'>Transaction Amount Over Time</div>", unsafe_allow_html=True)
    fig = px.line(df.sort_values("Time"), x="Time", y="Amount", color="Risk",
                  color_discrete_map={"High": "#D62839", "Low": "#00C985"})
    st.plotly_chart(fig, use_container_width=True)
