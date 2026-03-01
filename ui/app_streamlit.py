# app_streamlit.py
import os
import sys
from pathlib import Path
from contextlib import AbstractContextManager
from typing import cast

import joblib
import streamlit as st

# -----------------------------
# Repo root import
# -----------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data_load import load_data
from src.train import train_model
from src.config import DEFAULT_MODEL
from src.predict_user import predict_user

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Impact Forecaster", layout="wide")

# -----------------------------
# UI CSS
# -----------------------------
st.markdown(
    """
<style>
.stApp {
  background: radial-gradient(1200px 800px at 50% 10%, rgba(34,211,238,0.12), transparent 60%),
              radial-gradient(900px 600px at 80% 30%, rgba(132,204,22,0.10), transparent 60%),
              radial-gradient(1000px 700px at 20% 40%, rgba(59,130,246,0.10), transparent 60%),
              #070b14;
  color: #ffffff;
}
.block-container { padding-top: 2.2rem; max-width: 1100px; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* GRID BACKGROUND */
.stApp:before{
  content:"";
  position: fixed;
  inset: 0;
  pointer-events:none;
  opacity: 0.12;
  background-image:
    linear-gradient(to right, rgba(255,255,255,0.06) 1px, transparent 1px),
    linear-gradient(to bottom, rgba(255,255,255,0.06) 1px, transparent 1px);
  background-size: 72px 72px;
  mask-image: radial-gradient(circle at 50% 20%, black 35%, transparent 72%);
}

/* HERO */
.hero-wrap{
  padding: 34px 22px 26px 22px;
  border-radius: 24px;
  background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
  border: 1px solid rgba(255,255,255,0.10);
  box-shadow: 0 10px 35px rgba(0,0,0,0.35);
}
.pill{
  display:inline-flex;
  gap:10px;
  align-items:center;
  padding: 10px 16px;
  border-radius: 999px;
  border: 1px solid rgba(34,211,238,0.35);
  background: rgba(34,211,238,0.08);
  color: #ffffff;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  font-size: 14px;
}
.hero-title{
  margin: 20px 0 6px 0;
  font-size: 78px;
  line-height: 1.05;
  font-weight: 800;
  letter-spacing: -0.02em;
  text-align:center;
  color: #ffffff;
}
.grad{
  background: linear-gradient(90deg, #ffffff, #ffffff, #ffffff);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}
.hero-sub{
  margin-top: 14px;
  font-size: 26px;
  line-height: 1.5;
  color: #ffffff;
  text-align:center;
}
.hero-sub b{ color: #ffffff; }
.hero-sub .good{ color: #2dd4bf; font-weight: 800; }
.hero-sub .bad{ color: #ef4444; font-weight: 800; }
.feature-row{
  margin-top: 30px;
  display:flex;
  justify-content:center;
  gap: 46px;
  flex-wrap: wrap;
  color: #ffffff;
  font-size: 18px;
}
.feature-item{ display:flex; align-items:center; gap: 10px; }

/* BUTTONS */
div.stButton > button {
  border-radius: 16px !important;
  padding: 0.9rem 1.2rem !important;
  border: 1px solid rgba(34,211,238,0.35) !important;
  background: linear-gradient(90deg, rgba(34,211,238,0.25), rgba(163,230,53,0.20)) !important;
  color: #ffffff !important;
  font-weight: 700 !important;
  box-shadow: 0 10px 22px rgba(0,0,0,0.25);
  transition: transform 0.08s ease-in-out;
}
div.stButton > button:hover {
  transform: translateY(-1px);
  border: 1px solid rgba(34,211,238,0.55) !important;
}

/* FORM LABELS */
label {
  color: #ffffff !important;
  font-weight: 700 !important;
  font-size: 18px !important;
}

/* SELECTBOX */
div[data-baseweb="select"] {
  background-color: #111827 !important;
  border: 1px solid rgba(34,211,238,0.4) !important;
  border-radius: 14px !important;
}
div[data-baseweb="select"] span {
  color: #ffffff !important;
  font-size: 18px !important;
  font-weight: 600 !important;
}
div[role="listbox"] { background-color: #111827 !important; }
div[role="option"] { color: #ffffff !important; }

/* NUMBER INPUT */
div[data-baseweb="input"] {
  background-color: #111827 !important;
  border: 1px solid rgba(34,211,238,0.4) !important;
  border-radius: 14px !important;
}
input {
  color: #ffffff !important;
  background-color: #111827 !important;
  font-size: 18px !important;
  font-weight: 600 !important;
}

.small-muted{
  color: #ffffff;
  font-size: 14px;
  margin-top: 10px;
  text-align:center;
}
</style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Display labels ONLY (does NOT change dataset/model labels)
# -----------------------------
DISPLAY_LABELS = {
    "Class 1": "Improved",
    "Class 2": "Worsened",
    "Class 3": "Stayed Same",
    1: "Improved",
    2: "Worsened",
    3: "Stayed Same",
    "1": "Improved",
    "2": "Worsened",
    "3": "Stayed Same",
}

def pretty_label(x) -> str:
    return DISPLAY_LABELS.get(x, str(x))

# -----------------------------
# Model bundle: cache + optional artifacts
# -----------------------------
MODEL_VERSION = "v4-displaylabels-2026-02-28"
ENABLE_ARTIFACTS = os.environ.get("ENABLE_ARTIFACTS", "1") == "1"

ART_DIR = Path(ROOT_DIR) / "artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)

def artifact_path(model_name: str) -> Path:
    return ART_DIR / f"bundle_{model_name}_{MODEL_VERSION}.joblib"

def save_artifact(model_name: str, bundle: dict):
    if not ENABLE_ARTIFACTS:
        return
    joblib.dump(bundle, artifact_path(model_name))

def load_artifact(model_name: str):
    if not ENABLE_ARTIFACTS:
        return None
    p = artifact_path(model_name)
    if not p.exists():
        return None
    return joblib.load(p)

@st.cache_resource(show_spinner=False)
def get_model_bundle(model_name: str = DEFAULT_MODEL, version: str = MODEL_VERSION):
    bundle = load_artifact(model_name)
    if isinstance(bundle, dict) and "model" in bundle and "feature_names" in bundle:
        return bundle, "loaded"

    df = load_data()
    res = train_model(df, model_name=model_name)
    bundle = {"model": res.model, "feature_names": res.feature_names}
    save_artifact(model_name, bundle)
    return bundle, "trained"

with cast(AbstractContextManager[None], st.spinner("Loading model...")):
    bundle, MODEL_STATUS = get_model_bundle(DEFAULT_MODEL, MODEL_VERSION)

MODEL = bundle["model"]
FEATURE_NAMES = bundle["feature_names"]

# -----------------------------
# UI mappings
# -----------------------------
AGE_MAP = {
    "Under 25": 1,
    "25–34": 2,
    "35–44": 3,
    "45–54": 4,
    "55–64": 5,
    "65–79": 6,
    "80+": 7,
}

PROV_MAP = {
    "NL": 10, "PE": 11, "NS": 12, "NB": 13,
    "QC": 24, "ON": 35, "MB": 46,
    "SK": 47, "AB": 48, "BC": 59,
}

EDU_MAP = {
    "< High school": 1,
    "High school": 2,
    "Post-secondary": 3,
    "University": 4,
    "Prefer not to say": 9,
}

TENURE_MAP = {
    "Own outright": 1,
    "Own with mortgage": 2,
    "Rent": 3,
    "Prefer not to say": 9,
}

# -----------------------------
# Session state
# -----------------------------
STEPS = ["Overview", "Demographics", "Housing & Income", "Assets", "Debts", "Prediction"]

DEFAULT_PAYLOAD = {
    "Age Group": None,
    "Province of residence": None,
    "Education Level": None,
    "After-Tax Income": 0,
    "Home Ownership": None,
    "Mortgage Debt": 0,
    "Student Loan Debt": 0,
    "Credit Card Debt": 0,
    "Line of Credit Debt": 0,
    "Bank Deposits": 0,
    "TFSA Balance": 0,
}

if "step" not in st.session_state:
    st.session_state.step = 0

if "payload" not in st.session_state:
    st.session_state.payload = DEFAULT_PAYLOAD.copy()

if "prediction" not in st.session_state:
    st.session_state.prediction = None

payload = st.session_state.payload

def go_next():
    st.session_state.step = min(st.session_state.step + 1, len(STEPS) - 1)

def go_back():
    st.session_state.step = max(st.session_state.step - 1, 0)

def reset_all():
    st.session_state.step = 0
    st.session_state.payload = DEFAULT_PAYLOAD.copy()
    st.session_state.prediction = None
    st.rerun()

# -----------------------------
# Top progress
# -----------------------------
st.progress((st.session_state.step + 1) / len(STEPS))
st.divider()

# -----------------------------
# Step 0 (Fancy landing)
# -----------------------------
if st.session_state.step == 0:
    st.markdown(
        f"""
<div class="hero-wrap">
  <div style="text-align:center;">
    <span class="pill">⚡ ML-POWERED PREDICTIONS</span>
    <div class="hero-title">
      Predict Financial Impact of<br/>
      <span class="grad">Economic Shocks</span>
    </div>
    <div class="hero-sub">
      Enter your financial profile and get probability estimates of how a major<br/>
      economic shock could <span class="good">improve</span>, <b>stabilize</b>, or <span class="bad">worsen</span> your financial situation.
    </div>
    <div class="small-muted">Model: {DEFAULT_MODEL} ({MODEL_STATUS})</div>
  </div>

  <div class="feature-row">
    <div class="feature-item">📈 <span>Real-time Analysis</span></div>
    <div class="feature-item">🛡️ <span>Risk Assessment</span></div>
    <div class="feature-item">⚡ <span>Instant Predictions</span></div>
  </div>
</div>
        """,
        unsafe_allow_html=True
    )

    st.write("")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        st.button("Start →", type="primary", on_click=go_next)

# -----------------------------
# Step 1
# -----------------------------
elif st.session_state.step == 1:
    st.header("Demographics")

    age = st.selectbox("Age Group", ["Select..."] + list(AGE_MAP.keys()))
    payload["Age Group"] = None if age == "Select..." else AGE_MAP[age]

    prov = st.selectbox("Province", ["Select..."] + list(PROV_MAP.keys()))
    payload["Province of residence"] = None if prov == "Select..." else PROV_MAP[prov]

    edu = st.selectbox("Education Level", ["Select..."] + list(EDU_MAP.keys()))
    payload["Education Level"] = None if edu == "Select..." else EDU_MAP[edu]

# -----------------------------
# Step 2
# -----------------------------
elif st.session_state.step == 2:
    st.header("Housing & Income")

    tenure = st.selectbox("Home Ownership", ["Select..."] + list(TENURE_MAP.keys()))
    payload["Home Ownership"] = None if tenure == "Select..." else TENURE_MAP[tenure]

    payload["After-Tax Income"] = st.number_input(
        "After-Tax Income", min_value=0, value=int(payload["After-Tax Income"])
    )

    # Optional: structural rule for renting
    if payload["Home Ownership"] == 3:
        payload["Mortgage Debt"] = 0

# -----------------------------
# Step 3
# -----------------------------
elif st.session_state.step == 3:
    st.header("Assets")
    payload["Bank Deposits"] = st.number_input(
        "Bank Deposits", min_value=0, value=int(payload["Bank Deposits"])
    )
    payload["TFSA Balance"] = st.number_input(
        "TFSA Balance", min_value=0, value=int(payload["TFSA Balance"])
    )

# -----------------------------
# Step 4
# -----------------------------
elif st.session_state.step == 4:
    st.header("Debts")

    payload["Mortgage Debt"] = st.number_input(
        "Mortgage Debt", min_value=0, value=int(payload["Mortgage Debt"])
    )
    payload["Student Loan Debt"] = st.number_input(
        "Student Loan Debt", min_value=0, value=int(payload["Student Loan Debt"])
    )
    payload["Credit Card Debt"] = st.number_input(
        "Credit Card Debt", min_value=0, value=int(payload["Credit Card Debt"])
    )
    payload["Line of Credit Debt"] = st.number_input(
        "Line of Credit Debt", min_value=0, value=int(payload["Line of Credit Debt"])
    )

# -----------------------------
# Step 5 Prediction
# -----------------------------
else:
    st.header("Prediction")

    required = ["Age Group", "Province of residence", "Education Level", "Home Ownership"]
    missing = [k for k in required if payload.get(k) is None]

    if missing:
        st.warning("Missing required fields: " + ", ".join(missing))
    else:
        if st.button("Generate Prediction", type="primary"):
            out = predict_user(MODEL, FEATURE_NAMES, payload)
            st.session_state.prediction = {
                "predicted": out["most_likely"],
                "probabilities": out.get("probabilities", {}),
                "confidence": out.get("confidence", None),
            }

        if st.session_state.prediction:
            result = st.session_state.prediction
            st.subheader(f"Predicted: {pretty_label(result['predicted'])}")

            probs = result.get("probabilities", {})
            if probs:
                # show highest prob first
                for name, p in sorted(probs.items(), key=lambda x: -x[1]):
                    p = float(p)
                    st.write(f"{pretty_label(name)}: {p:.3f}")
                    st.progress(p)
            else:
                st.info("No probability output available for this model.")

    st.divider()
    st.button("Reset", on_click=reset_all)

# -----------------------------
# Bottom navigation
# -----------------------------
if st.session_state.step != 0:
    st.divider()
    b1, b2 = st.columns(2)

    with b1:
        st.button("← Back", on_click=go_back)

    if st.session_state.step < len(STEPS) - 1:
        with b2:
            st.button("Continue →", type="primary", on_click=go_next)