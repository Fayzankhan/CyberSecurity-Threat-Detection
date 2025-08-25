from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any
from pathlib import Path
import joblib
import json
import numpy as np
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from ..utils.columns import ALL_FEATURES
from ..config.settings import ARTIFACTS_DIR, IS_PRODUCTION, PORT, ALLOWED_HOSTS

ARTIFACTS = ARTIFACTS_DIR

class BatchEvents(BaseModel):
    records: List[Dict[str, Any]]

app = FastAPI(title="Cyber Threat Detector", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    """Health check endpoint"""
    try:
        return {"status": "ok", "message": "API is running"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.post("/predict-batch")
async def predict_batch(payload: BatchEvents):
    """Batch prediction endpoint with detailed error handling"""
    try:
        logger.info("Received prediction request")
        
        # Log input data shape
        logger.info(f"Received {len(payload.records)} records")
        
        # Create DataFrame
        try:
            df = pd.DataFrame(payload.records)
            logger.info(f"Created DataFrame with shape: {df.shape}")
        except Exception as e:
            logger.error(f"Failed to create DataFrame: {str(e)}")
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid data format: {str(e)}"}
            )

        # Validate columns
        missing = [c for c in ALL_FEATURES if c not in df.columns]
        if missing:
            logger.error(f"Missing columns: {missing}")
            return JSONResponse(
                status_code=400,
                content={"error": f"Missing columns: {missing}"}
            )

        # Create dummy predictions for testing
        logger.info("Generating predictions")
        try:
            # Generate random predictions for testing
            predictions = np.random.randint(0, 2, size=len(df)).tolist()
            probabilities = np.random.random(size=len(df)).tolist()
            
            logger.info("Successfully generated predictions")
            return {
                "predictions": predictions,
                "probabilities": probabilities
            }
        except Exception as e:
            logger.error(f"Prediction generation failed: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": f"Prediction failed: {str(e)}"}
            )

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Unexpected error: {str(e)}"}
        )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Global exception handler caught: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )