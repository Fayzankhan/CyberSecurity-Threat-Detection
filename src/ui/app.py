import time
import json
import sys
from pathlib import Path
import traceback

import requests
import pandas as pd
import streamlit as st
import plotly.express as px

# Add project root to Python path and print debug info
try:
    project_root = Path(__file__).resolve().parent.parent.parent
    st.write(f"Project root: {project_root}")
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
        st.write(f"Added to Python path: {project_root}")
    
    from src.utils.columns import ALL_FEATURES
    st.write("Successfully imported ALL_FEATURES")
except Exception as e:
    st.error(f"Import error: {str(e)}")
    st.code(traceback.format_exc())
    raise

st.set_page_config(page_title="Cyber Threat Detector", layout="wide")
ARTIFACTS = Path(__file__).resolve().parent.parent.parent / "artifacts"
METRICS_BIN = ARTIFACTS / "metrics.json"
METRICS_MULTI = ARTIFACTS / "metrics_multiclass.json"

from src.config import API_HOST

# Try to get API URL from Streamlit secrets (for cloud deployment)
try:
    API = st.secrets["api_url"]
    st.sidebar.success(f"Connected to API: {API}")
except Exception as e:
    API = API_HOST
    st.sidebar.warning(f"Using default API: {API}")

st.sidebar.title("🔐 Cyber Threat Detector")
page = st.sidebar.radio("Navigation", ["Dashboard", "Batch Prediction", "Live Demo"])

def load_metrics():
    bin_m, multi_m = None, None
    if METRICS_BIN.exists():
        with open(METRICS_BIN) as f:
            bin_m = json.load(f)
    if METRICS_MULTI.exists():
        with open(METRICS_MULTI) as f:
            multi_m = json.load(f)
    return bin_m, multi_m

def call_api(path: str, payload: dict):
    url = f"{API}{path}"
    r = requests.post(url, json=payload, timeout=10)
    r.raise_for_status()
    return r.json()

if page == "Dashboard":
    st.title("📊 Model Performance")
    bin_m, multi_m = load_metrics()
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Binary Model (Attack vs Normal)")
        if bin_m:
            st.metric("ROC-AUC", f"{bin_m.get('roc_auc', None):.3f}" if bin_m.get('roc_auc') is not None else "N/A")
            st.json(bin_m.get("classification_report", {}))
        else:
            st.warning("Run training: `python src/train.py`")

    with c2:
        st.subheader("Multiclass Model (dos/probe/r2l/u2r/normal)")
        if multi_m:
            st.json(multi_m.get("classification_report", {}))
        else:
            st.info("Multiclass metrics not found; training will generate them.")

elif page == "Batch Prediction":
    st.title("📂 Upload CSV and Detect Threats")
    uploaded = st.file_uploader("Upload CSV with the 41 feature columns", type=["csv"])
    model_type = st.radio("Model", ["Binary", "Multiclass"], horizontal=True)

    if uploaded:
        df = pd.read_csv(uploaded)
        st.write("Preview:", df.head())
        missing = [c for c in ALL_FEATURES if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            if st.button("🔍 Run Detection"):
                if model_type == "Binary":
                    res = call_api("/predict-batch", {"records": df[ALL_FEATURES].to_dict(orient="records")})
                    df["attack_probability"] = res["probabilities"]
                    df["is_attack"] = res["predictions"]
                    st.success("Done.")
                else:
                    res = call_api("/predict-multiclass", {"records": df[ALL_FEATURES].to_dict(orient="records")})
                    df["predicted_class"] = res["predictions"]
                    df["confidence"] = res["confidence"]
                    st.success("Done.")

                # Charts
                st.subheader("Threat Summary")
                if "is_attack" in df.columns:
                    pie_df = df["is_attack"].map({0:"normal",1:"attack"}).value_counts().reset_index()
                    pie_df.columns = ["class","count"]
                    fig = px.pie(pie_df, names="class", values="count", title="Attack vs Normal")
                    st.plotly_chart(fig, use_container_width=True)
                if "predicted_class" in df.columns:
                    bar_df = df["predicted_class"].value_counts().reset_index()
                    bar_df.columns = ["class","count"]
                    fig2 = px.bar(bar_df, x="class", y="count", title="Predicted Attack Types")
                    st.plotly_chart(fig2, use_container_width=True)

                st.subheader("Results")
                st.dataframe(df.head(100))

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Results", csv, "threat_results.csv", "text/csv")

elif page == "Live Demo":
    st.title("⚡ Live Threat Detection (NSL-KDD Test Stream)")

    rate = st.slider("Events per refresh", min_value=5, max_value=200, value=25, step=5)
    model_type = st.radio("Model", ["Binary", "Multiclass"], horizontal=True)

    if "live_idx" not in st.session_state:
        st.session_state.live_idx = 0
    if "live_df" not in st.session_state:
        # Load NSL-KDD test set from data/
        data_path = Path(__file__).resolve().parent.parent.parent / "data" / "KDDTest+.txt"
        from src.utils.columns import CSV_COLUMNS
        if not data_path.exists():
            st.warning("Dataset not found. Run: python src/train.py (it downloads data).")
        else:
            st.session_state.live_df = pd.read_csv(data_path, names=CSV_COLUMNS)

    if "live_df" in st.session_state and st.session_state.live_df is not None:
        df = st.session_state.live_df
        start = st.session_state.live_idx
        end = min(start + rate, len(df))
        batch = df.iloc[start:end]
        st.write(f"Processing events {start} → {end} / {len(df)}")

        if start < end:
            if model_type == "Binary":
                res = call_api("/predict-batch", {"records": batch[ALL_FEATURES].to_dict(orient='records')})
                batch_preds = pd.Series(res["predictions"], index=batch.index)
                batch_prob = pd.Series(res["probabilities"], index=batch.index)
                df.loc[batch.index, "is_attack"] = batch_preds
                df.loc[batch.index, "attack_probability"] = batch_prob
            else:
                res = call_api("/predict-multiclass", {"records": batch[ALL_FEATURES].to_dict(orient='records')})
                batch_cls = pd.Series(res["predictions"], index=batch.index)
                df.loc[batch.index, "predicted_class"] = batch_cls

            st.session_state.live_idx = end

        # KPIs
        total = st.session_state.live_idx
        attacks = int((df.loc[:end-1, "is_attack"]==1).sum()) if "is_attack" in df.columns else 0
        c1, c2, c3 = st.columns(3)
        c1.metric("Processed events", total)
        c2.metric("Detected attacks", attacks)
        rate_attacks = (attacks / total * 100.0) if total else 0.0
        c3.metric("Attack rate", f"{rate_attacks:.2f}%")

        # Charts
        if "is_attack" in df.columns:
            pie_df = df.loc[:end-1, "is_attack"].fillna(0).map({0:"normal",1:"attack"}).value_counts().reset_index()
            pie_df.columns = ["class","count"]
            fig = px.pie(pie_df, names="class", values="count", title="Attack vs Normal (live)")
            st.plotly_chart(fig, use_container_width=True)

        if "predicted_class" in df.columns:
            bar_df = df.loc[:end-1, "predicted_class"].dropna().value_counts().reset_index()
            bar_df.columns = ["class","count"]
            fig2 = px.bar(bar_df, x="class", y="count", title="Attack Types (live)")
            st.plotly_chart(fig2, use_container_width=True)

        # Control buttons
        colA, colB, colC = st.columns(3)
        if colA.button("Next batch ▶️"):
            st.experimental_rerun()
        if colB.button("Reset 🔄"):
            st.session_state.live_idx = 0
            if "is_attack" in df.columns:
                df["is_attack"] = None
                df["attack_probability"] = None
            if "predicted_class" in df.columns:
                df["predicted_class"] = None
            st.experimental_rerun()

    st.info("Tip: Keep the FastAPI server running (uvicorn) while using Live Demo.")
