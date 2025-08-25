import time
import json
import sys
from pathlib import Path
import traceback
import logging

import requests
import pandas as pd
import streamlit as st
import plotly.express as px

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit page first
try:
    st.set_page_config(
        page_title="Cyber Threat Detector",
        page_icon="🔐",
        layout="wide",
        initial_sidebar_state="expanded"
    )
except Exception as e:
    st.write("Error setting page config:", str(e))

# Add project root to Python path
try:
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    
    from src.utils.columns import ALL_FEATURES
    logger.info("Successfully imported ALL_FEATURES")
except Exception as e:
    st.error("⚠️ Setup Error")
    st.error(f"Error details: {str(e)}")
    st.code(traceback.format_exc())
    st.info("Please contact support if this error persists.")
    st.stop()

# Initialize paths
ARTIFACTS = Path(__file__).resolve().parent.parent.parent / "artifacts"
METRICS_BIN = ARTIFACTS / "metrics.json"
METRICS_MULTI = ARTIFACTS / "metrics_multiclass.json"

# Configure API
try:
    from src.config import API_HOST
    API = st.secrets.get("api_url", API_HOST)
    logger.info(f"Using API endpoint: {API}")
    
    # Test API connection
    response = requests.get(f"{API}/health")
    if response.status_code == 200:
        st.sidebar.success(f"✅ Connected to API: {API}")
    else:
        st.sidebar.warning(f"⚠️ API returned status code: {response.status_code}")
except Exception as e:
    logger.error(f"API connection error: {str(e)}")
    st.sidebar.error(f"❌ API connection failed: {str(e)}")
    API = API_HOST
    st.sidebar.info(f"Using fallback API: {API}")

# Main app UI
st.sidebar.title("🔐 Cyber Threat Detector")
page = st.sidebar.radio("Navigation", ["Dashboard", "Batch Prediction", "Live Demo"])

def load_metrics():
    """Load model metrics from files"""
    bin_m, multi_m = None, None
    try:
        if METRICS_BIN.exists():
            with open(METRICS_BIN) as f:
                bin_m = json.load(f)
        if METRICS_MULTI.exists():
            with open(METRICS_MULTI) as f:
                multi_m = json.load(f)
    except Exception as e:
        logger.error(f"Error loading metrics: {str(e)}")
        st.error(f"Failed to load metrics: {str(e)}")
    return bin_m, multi_m

def call_api(path: str, payload: dict):
    """Make API call with error handling"""
    try:
        url = f"{API}{path}"
        logger.info(f"Calling API: {url}")
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"API call failed: {str(e)}")
        st.error(f"API call failed: {str(e)}")
        raise

# Page content
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
            st.warning("Binary model metrics not available")

    with c2:
        st.subheader("Multiclass Model (dos/probe/r2l/u2r/normal)")
        if multi_m:
            st.json(multi_m.get("classification_report", {}))
        else:
            st.warning("Multiclass model metrics not available")

elif page == "Batch Prediction":
    st.title("📂 Upload CSV and Detect Threats")
    uploaded = st.file_uploader("Upload CSV with the 41 feature columns", type=["csv"])
    model_type = st.radio("Model", ["Binary", "Multiclass"], horizontal=True)

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.write("Preview:", df.head())
            missing = [c for c in ALL_FEATURES if c not in df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                if st.button("🔍 Run Detection"):
                    with st.spinner("Processing..."):
                        if model_type == "Binary":
                            res = call_api("/predict-batch", {"records": df[ALL_FEATURES].to_dict(orient="records")})
                            df["attack_probability"] = res["probabilities"]
                            df["is_attack"] = res["predictions"]
                        else:
                            res = call_api("/predict-multiclass", {"records": df[ALL_FEATURES].to_dict(orient="records")})
                            df["predicted_class"] = res["predictions"]
                            df["confidence"] = res["confidence"]
                        st.success("Analysis complete! 🎉")

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
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            st.error(f"Error processing file: {str(e)}")

elif page == "Live Demo":
    st.title("⚡ Live Threat Detection (NSL-KDD Test Stream)")

    rate = st.slider("Events per refresh", min_value=5, max_value=200, value=25, step=5)
    model_type = st.radio("Model", ["Binary", "Multiclass"], horizontal=True)

    if "live_idx" not in st.session_state:
        st.session_state.live_idx = 0
    if "live_df" not in st.session_state:
        try:
            # Try to download dataset if not present
            data_path = Path(__file__).resolve().parent.parent.parent / "data" / "KDDTest+.txt"
            if not data_path.exists():
                st.warning("Downloading dataset...")
                from src.download_data import main as download_data
                download_data()
                st.success("Dataset downloaded successfully!")
            
            from src.utils.columns import CSV_COLUMNS
            st.session_state.live_df = pd.read_csv(data_path, names=CSV_COLUMNS)
            st.success(f"Loaded {len(st.session_state.live_df)} records from dataset")
        except Exception as e:
            logger.error(f"Error loading test data: {str(e)}")
            st.error(f"Error loading test data: {str(e)}")

    if "live_df" in st.session_state and st.session_state.live_df is not None:
        df = st.session_state.live_df
        start = st.session_state.live_idx
        end = min(start + rate, len(df))
        batch = df.iloc[start:end]
        st.write(f"Processing events {start} → {end} / {len(df)}")

        if start < end:
            try:
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
            except Exception as e:
                logger.error(f"Error in prediction: {str(e)}")
                st.error(f"Error in prediction: {str(e)}")

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

    st.info("Tip: Keep the FastAPI server running while using Live Demo.")