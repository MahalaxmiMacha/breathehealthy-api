
import os
import json
import joblib
import numpy as np
import tensorflow as tf
import xgboost as xgb
import shap

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any


# ============================================================
# App
# ============================================================

app = FastAPI(
    title="BreatheHealthy AI API",
    description="Hybrid LSTM + XGBoost Air Quality Forecasting API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")


# ============================================================
# Load models
# ============================================================

print("Loading BreatheHealthy models...")

LSTM_PATH = os.path.join(MODEL_DIR, "lstm_model_final.keras")
XGB_PATH = os.path.join(MODEL_DIR, "xgb_model.json")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_final.pkl")
META_PATH = os.path.join(MODEL_DIR, "meta_model_final.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "features_final.json")

lstm_model = tf.keras.models.load_model(LSTM_PATH)

xgb_model = xgb.XGBRegressor()
xgb_model.load_model(XGB_PATH)

scaler = joblib.load(SCALER_PATH)
meta_model = joblib.load(META_PATH)

with open(FEATURES_PATH, "r") as f:
    features_data = json.load(f)

if isinstance(features_data, dict):
    FEATURES = features_data.get("features", features_data.get("FEATURES"))
else:
    FEATURES = features_data

if not FEATURES:
    raise RuntimeError("Could not load feature list.")

SEQ_LEN = 24

print("Models loaded successfully.")
print("Features:", len(FEATURES))
print("Sequence length:", SEQ_LEN)


# ============================================================
# Schemas
# ============================================================

class PredictionRequest(BaseModel):
    records: List[Dict[str, Any]]


# ============================================================
# AQI category
# ============================================================

def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


# ============================================================
# Validate records
# ============================================================

def prepare_records(records):

    if len(records) != SEQ_LEN:
        raise HTTPException(
            status_code=400,
            detail=f"Exactly {SEQ_LEN} consecutive hourly records are required."
        )

    try:
        matrix = []

        for record in records:
            row = []

            for feature in FEATURES:
                if feature not in record:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Missing feature: {feature}"
                    )

                value = float(record[feature])

                if not np.isfinite(value):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid value for feature: {feature}"
                    )

                row.append(value)

            matrix.append(row)

        X_raw = np.asarray(matrix, dtype=np.float32)

        return X_raw

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input data: {str(e)}"
        )


# ============================================================
# Health
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "online",
        "service": "BreatheHealthy AI API",
        "models": {
            "xgboost": True,
            "lstm": True,
            "stacked_hybrid": True,
            "shap": True
        },
        "features": len(FEATURES),
        "sequence_length": SEQ_LEN,
        "prediction_target": "Next-hour US AQI"
    }


# ============================================================
# Prediction
# ============================================================

@app.post("/predict")
def predict(request: PredictionRequest):

    X_raw = prepare_records(request.records)

    # Scale input
    X_scaled = scaler.transform(X_raw)

    # LSTM
    X_lstm = X_scaled.reshape(1, SEQ_LEN, len(FEATURES))

    lstm_prediction = float(
        lstm_model.predict(X_lstm, verbose=0)[0][0]
    )

    # XGBoost uses latest hour
    X_xgb = X_scaled[-1].reshape(1, -1)

    xgb_prediction = float(
        xgb_model.predict(X_xgb)[0]
    )

    # Hybrid meta learner
    meta_input = np.array(
        [[lstm_prediction, xgb_prediction]],
        dtype=np.float32
    )

    hybrid_prediction = float(
        meta_model.predict(meta_input)[0]
    )

    return {
        "success": True,
        "prediction_target": "Next-hour US AQI",
        "predicted_aqi": round(hybrid_prediction, 2),
        "lstm_prediction": round(lstm_prediction, 2),
        "xgboost_prediction": round(xgb_prediction, 2),
        "category": get_aqi_category(hybrid_prediction),
        "input_hours": SEQ_LEN,
        "features": len(FEATURES)
    }


# ============================================================
# SHAP Explanation
# ============================================================

@app.post("/explain")
def explain(request: PredictionRequest):

    X_raw = prepare_records(request.records)

    X_scaled = scaler.transform(X_raw)

    # Explain latest hour using XGBoost
    X_latest = X_scaled[-1].reshape(1, -1)

    explainer = shap.TreeExplainer(xgb_model)

    shap_values = explainer.shap_values(X_latest)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_row = np.asarray(shap_values)[0]

    base_value = explainer.expected_value

    if isinstance(base_value, np.ndarray):
        base_value = float(base_value.reshape(-1)[0])
    else:
        base_value = float(base_value)

    factors = []

    for i, feature in enumerate(FEATURES):

        shap_value = float(shap_row[i])
        raw_value = float(X_raw[-1][i])

        factors.append({
            "feature": feature,
            "value": round(raw_value, 4),
            "shap_value": round(shap_value, 4),
            "impact": "increases" if shap_value > 0 else "decreases"
        })

    factors.sort(
        key=lambda x: abs(x["shap_value"]),
        reverse=True
    )

    return {
        "success": True,
        "explanation_target": "Next-hour AQI",
        "base_value": round(base_value, 3),
        "top_factors": factors[:10],
        "total_features": len(FEATURES)
    }


# ============================================================
# Root
# ============================================================

@app.get("/")
def root():

    return {
        "service": "BreatheHealthy AI API",
        "status": "online",
        "message": "Hybrid LSTM + XGBoost AQI forecasting service",
        "docs": "/docs",
        "health": "/health"
    }
