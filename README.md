# Cyber Threat Detector

A machine learning-based cyber threat detection system using the NSL-KDD dataset. The application includes both binary (attack vs. normal) and multiclass (dos/probe/r2l/u2r/normal) classification models.

## Features

- 🔍 Real-time threat detection
- 📊 Interactive dashboard with model performance metrics
- 📂 Batch prediction support
- ⚡ Live demo with streaming data
- 🎯 Both binary and multiclass classification

## Setup Instructions

1. **Create and activate virtual environment**
   ```bash
   # Create virtual environment
   python3 -m venv venv
   
   # Activate virtual environment
   source venv/bin/activate
   ```

2. **Install the package in development mode**
   ```bash
   # Install in editable mode with all dependencies
   pip install -e .
   ```

3. **Download dataset and train models**
   ```bash
   # Download NSL-KDD dataset and train both binary and multiclass models
   python -m src.train
   ```
   This will:
   - Download the NSL-KDD dataset (KDDTrain+.txt and KDDTest+.txt)
   - Train the binary classification model
   - Train the multiclass classification model
   - Save models and metrics in the artifacts directory

## Running the Application

### Local Development

The application consists of two components that need to run simultaneously:

1. **Start the FastAPI server**
   ```bash
   # Start the API server
   uvicorn src.app.api:app --reload --port 8000
   ```

2. **Launch the Streamlit UI** (in a new terminal)
   ```bash
   # Activate virtual environment in the new terminal
   source venv/bin/activate
   
   # Start the Streamlit application
   streamlit run src/ui/app.py
   ```

The UI will be available at:
- Local URL: http://localhost:8502
- Network URL: http://[your-ip]:8502

### Production Deployment

The application is deployed and available at:
- Frontend (Streamlit): https://cybersecurity-threat-detection-pvtnkqz3ijezo22xeadxvj.streamlit.app
- Backend API (Render): https://cyber-threat-detector-api.onrender.com

To deploy your own instance:

1. **Deploy the FastAPI Backend**:
   - Fork this repository
   - Sign up on [Render.com](https://render.com)
   - Create a new Web Service (or use **Blueprint** from `render.yaml` so settings stay in sync)
   - Connect your repository
   - Configure:
     - **Build Command** (required for PyTorch / “CNN/LSTM” in the UI — do not use `pip install` only):
       ```bash
       bash scripts/render_build.sh
       ```
       This installs dependencies only (lean build for Render free tier stability). The file `.render-buildpacks.json` must **not** override this with a conflicting command.
     - **Start Command**:
       ```bash
       gunicorn src.app.api:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --timeout 120
       ```
     - Environment Variables:
       - `ENVIRONMENT`: `production`
   - If the service already exists: open **Settings → Build & Deploy**, paste the build command above, then **Manual Deploy → Clear build cache & deploy**.
   - After deploy, open `https://<your-service>.onrender.com/health`. On free tier, deep models may remain unavailable and the API will use sklearn fallback to keep requests stable.
   - Deep-learning checkpoints (`model_dl_*.pt`, `preprocess_dl_*.joblib`, `metrics_dl_*.json`) are **committed in this repo** so `/health` shows them as available on Render without training at deploy time. To refresh them: `python -m src.train_dl --quick`, then commit the updated files.

2. **Deploy the Streamlit Frontend**:
   - Go to [Streamlit Cloud](https://share.streamlit.app)
   - Connect your repository
   - Configure:
     - Main file path: `src/ui/app.py`
     - Python version: `3.10.12`
     - Add secret:
       - Key: `api_url`
       - Value: Your Render API URL

## Application Components

### 1. Dashboard
- View model performance metrics
- ROC-AUC scores
- Classification reports

### 2. Batch Prediction
- Upload CSV files with 41 feature columns
- Get predictions for multiple records
- Visualize results with charts
- Download predictions as CSV

### 3. Live Demo
- Stream NSL-KDD test data
- Real-time threat detection
- Interactive visualization
- Adjustable streaming rate

## Project Structure

```
cyber-threat-detector/
├── artifacts/           # Trained models and metrics
├── data/               # Dataset files
├── src/
│   ├── app/           # FastAPI backend
│   ├── ui/            # Streamlit frontend
│   ├── utils/         # Shared utilities
│   └── train.py       # Training script
└── setup.py           # Package configuration
```

## Requirements

- Python 3.10+
- FastAPI
- Streamlit
- Pandas
- Plotly
- Scikit-learn
- Other dependencies (see setup.py)

## Notes

- Keep the FastAPI server running while using the Live Demo feature
- The application expects the NSL-KDD dataset in the data directory
- Models and metrics are stored in the artifacts directory
- Use the binary model for simple attack detection
- Use the multiclass model for detailed attack classification