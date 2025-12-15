import streamlit as st
import base64

st.set_page_config(page_title="Fraud Detection System", layout="wide")

# Load CSS
with open("styles/theme.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Hide sidebar & header for landing
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {display: none;}
    header[data-testid="stHeader"] {background: transparent;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Background video function
def video_background(video_path: str):
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    encoded = base64.b64encode(video_bytes).decode()
    st.markdown(
        f"""
        <video id="bg-video" autoplay loop muted playsinline>
            <source src="data:video/mp4;base64,{encoded}" type="video/mp4">
        </video>
        """,
        unsafe_allow_html=True,
    )

video_background("assets/14250435_1920_1080_30fps.mp4")

# Landing center
st.markdown(
    """
    <div class="landing-container">
        <div class="title-center">Credit Card Fraud Detection Dashboard</div>
        <p class="subtitle-center">
            Real-time monitoring • AI-powered detection • Deep analytics
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Navigation cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("Overview", use_container_width=True):
        st.switch_page("pages/Overview.py")
with col2:
    if st.button("Prediction", use_container_width=True):
        st.switch_page("pages/Predict.py")
with col3:
    if st.button("Analytics", use_container_width=True):
        st.switch_page("pages/Analytics.py")
with col4:
    if st.button("Performance", use_container_width=True):
        st.switch_page("pages/Model_Performance.py")
