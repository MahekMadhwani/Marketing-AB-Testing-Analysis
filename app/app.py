"""
Streamlit app: Marketing A/B Test - Conversion Predictor
Serves the XGBoost model trained in Step 6 of the analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Ad Conversion Predictor", page_icon="📊", layout="centered")

# =========================================================
# Load model (cached so it only loads once per session)
# Path is built relative to THIS file's location, not the
# working directory - this matters because Streamlit Cloud
# runs from the repo root, not from inside the app/ subfolder.
# =========================================================
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xgb_model.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

FEATURE_ORDER = [
    "is_ad", "log_total_ads", "hour_sin", "hour_cos",
    "day_Monday", "day_Saturday", "day_Sunday", "day_Thursday", "day_Tuesday", "day_Wednesday"
]

def build_feature_vector(is_ad, total_ads, hour, day):
    log_total_ads = np.log1p(total_ads)
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    # one-hot day encoding (Friday is the baseline/dropped category)
    day_flags = {f"day_{d}": 0 for d in ["Monday", "Saturday", "Sunday", "Thursday", "Tuesday", "Wednesday"]}
    if day != "Friday":
        day_flags[f"day_{day}"] = 1

    row = {
        "is_ad": int(is_ad),
        "log_total_ads": log_total_ads,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        **day_flags
    }
    return pd.DataFrame([row])[FEATURE_ORDER]

# =========================================================
# UI
# =========================================================
st.title("📊 Marketing A/B Test — Conversion Predictor")
st.markdown(
    "This app is powered by an XGBoost model trained on 588,101 real users from a "
    "marketing A/B test (ad vs. PSA). Adjust the inputs below to see the model's "
    "predicted conversion probability. "
    "[See the full analysis notebook →](https://github.com/YOUR-USERNAME/YOUR-REPO)"
)

st.divider()

col1, col2 = st.columns(2)
with col1:
    group = st.radio("Test group", ["ad", "psa"], help="Was the user shown an ad or a PSA?")
    total_ads = st.slider("Total ads/PSAs seen", min_value=1, max_value=500, value=20)

with col2:
    day = st.selectbox("Day of week (most exposure)", 
                        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    hour = st.slider("Hour of day (most exposure)", min_value=0, max_value=23, value=16)

is_ad = 1 if group == "ad" else 0

# =========================================================
# Predict
# =========================================================
X = build_feature_vector(is_ad, total_ads, hour, day)
prob = model.predict_proba(X)[0, 1]

st.divider()
st.subheader("Predicted conversion probability")

st.metric(label="", value=f"{prob*100:.2f}%")
st.progress(min(float(prob) * 5, 1.0))  # scaled x5 since real rates are small (~2.5%), for visibility

st.caption(
    "Note: baseline conversion rate in the training data was ~2.5%, so even a doubling or "
    "tripling of that (e.g. 5-8%) represents a strong signal, not a small number."
)

with st.expander("What's happening under the hood?"):
    st.markdown("""
    - The model is an **XGBoost classifier** trained on real experiment data (see the full project notebook)
    - Features used: whether the user saw an ad vs PSA, log-transformed total ad exposure, and cyclically-encoded day/hour
    - **`total_ads` is by far the strongest predictor** — SHAP analysis in the full project shows this clearly
    - This tool is for demonstration; see the [business write-up](https://github.com/YOUR-USERNAME/YOUR-REPO/blob/main/business_writeup.md) for the actual A/B test recommendation, which does not rely on this predictive model
    """)

st.divider()
st.caption("Built with Streamlit · Model: XGBoost · Data: [Kaggle Marketing A/B Testing](https://www.kaggle.com/datasets/faviovaz/marketing-ab-testing/data)")
