import os
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

_UI_DIR = Path(__file__).resolve().parent
_FAVICON = _UI_DIR / "favicon.svg"

# Minimal stroke icons (avoid emoji; consistent with dashboard UIs)
_S = '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">'
ICONS = {
    "shield": _S + '<path d="M12 3 4 7v5c0 5 3.5 9 8 10 4.5-1 8-5 8-10V7l-8-4Z"/></svg>',
    "chart": _S + '<path d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z"/></svg>',
    "folder": _S + '<path d="M2.25 12.75V12A2.25 2.25 0 014.5 9.75h15A2.25 2.25 0 0121.75 12v.75m-8.69-6.44-2.12-2.12a1.5 1.5 0 00-1.061-.44H4.5A2.25 2.25 0 002.25 6v12a2.25 2.25 0 002.25 2.25h15A2.25 2.25 0 0021.75 18V9a2.25 2.25 0 00-2.25-2.25h-5.379a1.5 1.5 0 01-1.06-.44z"/></svg>',
    "bolt": _S + '<path d="M3.75 13.5l10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75z"/></svg>',
}


def page_heading(text: str, icon_key: str) -> None:
    svg = ICONS.get(icon_key, ICONS["shield"])
    st.markdown(
        f'<h1 class="page-h1">{svg}<span>{text}</span></h1>',
        unsafe_allow_html=True,
    )


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable file watcher to avoid inotify limits
import os
os.environ['STREAMLIT_SERVER_WATCH_DIRS'] = 'false'

# Configure Streamlit page
try:
    st.set_page_config(
        page_title="Cyber Threat Detector",
        page_icon=str(_FAVICON) if _FAVICON.exists() else None,
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'About': '# Cyber Threat Detector\nA machine learning-based cyber threat detection system.'
        }
    )
except Exception as e:
    st.write("Error setting page config:", str(e))

# Add custom CSS for mobile responsiveness
st.markdown("""
    <style>
        .page-h1 {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            font-size: clamp(1.5rem, 4vw, 2rem);
            font-weight: 600;
            margin: 0 0 0.75rem 0;
            line-height: 1.2;
            color: inherit;
        }
        .page-h1 svg { flex-shrink: 0; width: 1.75rem; height: 1.75rem; opacity: 0.92; }
        .ui-brand {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-weight: 600;
            font-size: 1.15rem;
            margin-bottom: 1rem;
            color: inherit;
        }
        .ui-brand svg { flex-shrink: 0; width: 1.35rem; height: 1.35rem; opacity: 0.9; }
        /* Mobile-first responsive design */
        @media screen and (max-width: 768px) {
            /* Main content padding */
            .main .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
                padding-left: 1rem;
                padding-right: 1rem;
            }
            
            /* Sidebar adjustments */
            .css-1d391kg {
                padding-top: 1rem;
            }
            
            /* Title font size */
            h1 {
                font-size: 1.75rem !important;
            }
            
            h2, h3 {
                font-size: 1.25rem !important;
            }
            
            /* Metric cards - stack vertically on mobile */
            [data-testid="stMetricValue"] {
                font-size: 1.5rem !important;
            }
            
            [data-testid="stMetricLabel"] {
                font-size: 0.9rem !important;
            }
            
            /* Buttons - full width on mobile */
            .stButton > button {
                width: 100%;
                margin-bottom: 0.5rem;
            }
            
            /* File uploader */
            .uploadedFile {
                font-size: 0.9rem;
            }
            
            /* Dataframe container - horizontal scroll */
            .dataframe {
                font-size: 0.8rem;
                overflow-x: auto;
                display: block;
            }
            
            /* Radio buttons - stack vertically on mobile */
            .stRadio > div {
                flex-direction: column;
            }
            
            .stRadio > div > label {
                margin-bottom: 0.5rem;
            }
            
            /* Slider adjustments */
            .stSlider {
                margin-bottom: 1rem;
            }
            
            /* Chart containers */
            .js-plotly-plot {
                width: 100% !important;
            }
            
            /* JSON display */
            .stJson {
                font-size: 0.75rem;
                overflow-x: auto;
            }
            
            /* Info/warning/error messages */
            .stAlert {
                font-size: 0.9rem;
                padding: 0.75rem;
            }
            
            /* Sidebar title */
            .css-1v0mbdj {
                font-size: 1.25rem !important;
            }
        }
        
        /* Tablet adjustments */
        @media screen and (min-width: 769px) and (max-width: 1024px) {
            .main .block-container {
                padding-left: 2rem;
                padding-right: 2rem;
            }
        }
        
        /* Ensure tables are scrollable */
        .dataframe-container {
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
        }
        
        /* Better spacing for columns on mobile */
        @media screen and (max-width: 768px) {
            [data-testid="column"] {
                padding: 0.5rem;
            }
        }
    </style>
""", unsafe_allow_html=True)

# Add project root to Python path
try:
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    
    from src.utils.columns import ALL_FEATURES
    logger.info("Successfully imported ALL_FEATURES")
except Exception as e:
    st.error("Setup error")
    st.error(f"Error details: {str(e)}")
    st.code(traceback.format_exc())
    st.info("Please contact support if this error persists.")
    st.stop()

# Initialize paths
ARTIFACTS = Path(__file__).resolve().parent.parent.parent / "artifacts"
METRICS_BIN = ARTIFACTS / "metrics.json"
METRICS_MULTI = ARTIFACTS / "metrics_multiclass.json"
METRICS_DL_BIN = ARTIFACTS / "metrics_dl_binary.json"
METRICS_DL_MULTI = ARTIFACTS / "metrics_dl_multiclass.json"

# Configure API
try:
    from src.config import API_HOST
    API = st.secrets.get("api_url", API_HOST)
    logger.info(f"Using API endpoint: {API}")
    
    # Test API connection
    response = requests.get(f"{API}/health")
    if response.status_code == 200:
        st.sidebar.success(f"Connected to API: {API}")
    else:
        st.sidebar.warning(f"API returned status code: {response.status_code}")
except Exception as e:
    logger.error(f"API connection error: {str(e)}")
    st.sidebar.error(f"API connection failed: {str(e)}")
    API = API_HOST
    st.sidebar.info(f"Using fallback API: {API}")

# Main app UI
st.sidebar.markdown(
    f'<div class="ui-brand">{ICONS["shield"]}<span>Cyber Threat Detector</span></div>',
    unsafe_allow_html=True,
)
page = st.sidebar.radio("Navigation", ["Dashboard", "Batch Prediction", "Live Demo"])

def load_metrics():
    """Load model metrics from files (sklearn + optional PyTorch CNN/LSTM)."""
    bin_m, multi_m, dl_bin_m, dl_multi_m = None, None, None, None
    try:
        if METRICS_BIN.exists():
            with open(METRICS_BIN) as f:
                bin_m = json.load(f)
        if METRICS_MULTI.exists():
            with open(METRICS_MULTI) as f:
                multi_m = json.load(f)
        if METRICS_DL_BIN.exists():
            with open(METRICS_DL_BIN) as f:
                dl_bin_m = json.load(f)
        if METRICS_DL_MULTI.exists():
            with open(METRICS_DL_MULTI) as f:
                dl_multi_m = json.load(f)
    except Exception as e:
        logger.error(f"Error loading metrics: {str(e)}")
        st.error(f"Failed to load metrics: {str(e)}")
    return bin_m, multi_m, dl_bin_m, dl_multi_m


def prediction_payload(records: list, backend_choice: str) -> dict:
    mb = "deep" if backend_choice.strip().startswith("PyTorch") else "sklearn"
    return {"records": records, "model_backend": mb}


def _api_auth_headers() -> dict:
    """Bearer token from Streamlit secrets or env when API_SECRET_KEY is configured on the server."""
    key = None
    try:
        key = st.secrets.get("api_key")
    except Exception:
        pass
    if not key:
        key = os.environ.get("API_SECRET_KEY") or os.environ.get("API_KEY")
    if key:
        return {"Authorization": f"Bearer {key.strip()}"}
    return {}


def call_api(path: str, payload: dict):
    """Make API call with detailed error handling"""
    try:
        url = f"{API}{path}"
        st.info(f"Making API request to: {url}")
        
        # Make the request with increased timeout
        r = requests.post(url, json=payload, headers=_api_auth_headers(), timeout=30)
        
        # Check if we got a JSON response
        try:
            response_data = r.json()
        except Exception as e:
            st.error(f"Failed to parse API response as JSON: {str(e)}")
            st.code(r.text)  # Show raw response
            raise
        
        # Check for error in response
        if "error" in response_data:
            st.error(f"API Error: {response_data['error']}")
            if "detail" in response_data:
                st.code(response_data["detail"])
            raise Exception(response_data["error"])
        
        # If we got here, request was successful
        st.success("API request successful!")
        return response_data
        
    except requests.exceptions.Timeout:
        st.error("API request timed out. Please try again.")
        raise
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to API. Please check if the API server is running.")
        raise
    except Exception as e:
        st.error(f"API request failed: {str(e)}")
        raise

# Page content
if page == "Dashboard":
    page_heading("Model performance", "chart")
    bin_m, multi_m, dl_bin_m, dl_multi_m = load_metrics()
    st.caption(
        "Sklearn = RandomForest on one-hot + passthrough. Deep = 1D-CNN or LSTM over the "
        "preprocessed vector (one-hot + scaled numerics). Train: `python -m src.train_dl` or `--arch lstm`."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Binary — RandomForest")
        if bin_m:
            st.metric("ROC-AUC", f"{bin_m.get('roc_auc', None):.3f}" if bin_m.get('roc_auc') is not None else "N/A")
            st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
            st.json(bin_m.get("classification_report", {}))
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("Metrics not available (run `python -m src.train`).")

    with c2:
        st.subheader("Multiclass — RandomForest")
        if multi_m:
            st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
            st.json(multi_m.get("classification_report", {}))
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("Metrics not available (run `python -m src.train`).")

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Binary — PyTorch (CNN / LSTM)")
        if dl_bin_m:
            st.metric("ROC-AUC", f"{dl_bin_m.get('roc_auc', None):.3f}" if dl_bin_m.get('roc_auc') is not None else "N/A")
            st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
            st.json(dl_bin_m.get("classification_report", {}))
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Train with `python -m src.train_dl` to generate deep learning metrics.")

    with c4:
        st.subheader("Multiclass — PyTorch (CNN / LSTM)")
        if dl_multi_m:
            st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
            st.json(dl_multi_m.get("classification_report", {}))
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Train with `python -m src.train_dl` to generate deep learning metrics.")

elif page == "Batch Prediction":
    page_heading("Upload CSV and detect threats", "folder")
    uploaded = st.file_uploader("Upload CSV with the 41 feature columns", type=["csv"])
    model_type = st.radio("Model", ["Binary", "Multiclass"], horizontal=True)
    backend = st.radio(
        "Model backend",
        ["RandomForest (sklearn)", "PyTorch (CNN/LSTM)"],
        horizontal=True,
        help="Deep models require `python -m src.train_dl` (default CNN) or `--arch lstm`.",
    )

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.write("Preview:")
            # Make preview scrollable on mobile
            st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
            st.dataframe(df.head(), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            missing = [c for c in ALL_FEATURES if c not in df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                if st.button("Run detection"):
                    with st.spinner("Processing..."):
                        if model_type == "Binary":
                            res = call_api(
                                "/predict-batch",
                                prediction_payload(df[ALL_FEATURES].to_dict(orient="records"), backend),
                            )
                            df["attack_probability"] = res["probabilities"]
                            df["is_attack"] = res["predictions"]
                        else:
                            res = call_api(
                                "/predict-multiclass",
                                prediction_payload(df[ALL_FEATURES].to_dict(orient="records"), backend),
                            )
                            df["predicted_class"] = res["predictions"]
                            df["confidence"] = res["confidence"]
                        st.success("Analysis complete.")

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
                    # Make dataframe scrollable on mobile
                    st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                    st.dataframe(df.head(100), use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download results", csv, "threat_results.csv", "text/csv")
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            st.error(f"Error processing file: {str(e)}")

elif page == "Live Demo":
    page_heading("Live threat detection (NSL-KDD test stream)", "bolt")

    rate = st.slider("Events per refresh", min_value=5, max_value=200, value=25, step=5)
    model_type = st.radio("Model", ["Binary", "Multiclass"], horizontal=True)
    backend = st.radio(
        "Model backend",
        ["RandomForest (sklearn)", "PyTorch (CNN/LSTM)"],
        horizontal=True,
        help="Switching backend: use Reset stream to avoid mixing scores. Deep models need train_dl artifacts.",
    )

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
                    res = call_api(
                        "/predict-batch",
                        prediction_payload(batch[ALL_FEATURES].to_dict(orient="records"), backend),
                    )
                    batch_preds = pd.Series(res["predictions"], index=batch.index)
                    batch_prob = pd.Series(res["probabilities"], index=batch.index)
                    df.loc[batch.index, "is_attack"] = batch_preds
                    df.loc[batch.index, "attack_probability"] = batch_prob
                else:
                    res = call_api(
                        "/predict-multiclass",
                        prediction_payload(batch[ALL_FEATURES].to_dict(orient="records"), backend),
                    )
                    batch_cls = pd.Series(res["predictions"], index=batch.index)
                    df.loc[batch.index, "predicted_class"] = batch_cls

                st.session_state.live_idx = end
            except Exception as e:
                logger.error(f"Error in prediction: {str(e)}")
                st.error(f"Error in prediction: {str(e)}")

        # KPIs - responsive columns (will stack on mobile)
        total = st.session_state.live_idx
        attacks = int((df.loc[:end-1, "is_attack"]==1).sum()) if "is_attack" in df.columns else 0
        rate_attacks = (attacks / total * 100.0) if total else 0.0
        # Use columns that stack on mobile
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Processed events", total)
        with c2:
            st.metric("Detected attacks", attacks)
        with c3:
            st.metric("Attack rate", f"{rate_attacks:.2f}%")

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

        # Control buttons - responsive layout
        colA, colB, colC = st.columns([1, 1, 1])
        with colA:
            if st.button("Next batch", use_container_width=True):
                st.rerun()
        with colB:
            if st.button("Reset stream", use_container_width=True):
                st.session_state.live_idx = 0
                if "is_attack" in df.columns:
                    df["is_attack"] = None
                    df["attack_probability"] = None
                if "predicted_class" in df.columns:
                    df["predicted_class"] = None
                st.rerun()

    st.info(
        "Tip: Keep the FastAPI server running. For IoT-style micro-batches, use the WebSocket "
        "`/ws/predict` (see `python -m src.stream_simulator`)."
    )