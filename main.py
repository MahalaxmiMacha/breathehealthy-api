
import os

# ============================================================
# RENDER / LOW-MEMORY OPTIMIZATION
# Set these BEFORE importing TensorFlow, NumPy, or XGBoost.
# ============================================================

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import logging
from functools import lru_cache
from typing import Any, Dict, List

import joblib
import numpy as np
import tensorflow as tf
import xgboost as xgb

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("breathehealthy")

# ============================================================
# TENSORFLOW THREAD LIMIT
# ============================================================

try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    logger.warning("TensorFlow thread limits could not be changed.")

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="BreatheHealthy AI API",
    description="Hybrid LSTM + XGBoost Air Quality Forecasting API",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

LSTM_PATH = os.path.join(MODEL_DIR, "lstm_model_final.keras")
XGB_PATH = os.path.join(MODEL_DIR, "xgb_model.json")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_final.pkl")
META_PATH = os.path.join(MODEL_DIR, "meta_model_final.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "features_final.json")

SEQ_LEN = 24

# ============================================================
# LOAD TRAINED MODELS
# ============================================================

logger.info("Loading BreatheHealthy models...")

required_files = [
    LSTM_PATH,
    XGB_PATH,
    SCALER_PATH,
    META_PATH,
    FEATURES_PATH,
]

missing_files = [path for path in required_files if not os.path.isfile(path)]

if missing_files:
    raise RuntimeError(
        "Required model files are missing: "
        + ", ".join(os.path.basename(p) for p in missing_files)
    )

lstm_model = tf.keras.models.load_model(
    LSTM_PATH,
    compile=False,
)

xgb_model = xgb.XGBRegressor()
xgb_model.load_model(XGB_PATH)

try:
    xgb_model.set_params(n_jobs=1)
except Exception:
    pass

scaler = joblib.load(SCALER_PATH)
meta_model = joblib.load(META_PATH)

with open(FEATURES_PATH, "r", encoding="utf-8") as file:
    features_data = json.load(file)

if isinstance(features_data, dict):
    FEATURES = features_data.get(
        "features",
        features_data.get("FEATURES"),
    )
else:
    FEATURES = features_data

if not isinstance(FEATURES, list) or not FEATURES:
    raise RuntimeError(
        "features_final.json must contain a non-empty feature list."
    )

if not all(isinstance(feature, str) and feature for feature in FEATURES):
    raise RuntimeError("The feature list contains invalid feature names.")

if len(set(FEATURES)) != len(FEATURES):
    raise RuntimeError("The feature list contains duplicate names.")

NUM_FEATURES = len(FEATURES)

# Validate the scaler feature count where the object exposes it.
scaler_features = getattr(scaler, "n_features_in_", None)
if scaler_features is not None and scaler_features != NUM_FEATURES:
    raise RuntimeError(
        f"Scaler expects {scaler_features} features, "
        f"but features_final.json contains {NUM_FEATURES}."
    )

logger.info("Models loaded successfully.")
logger.info("Feature count: %s", NUM_FEATURES)
logger.info("Sequence length: %s", SEQ_LEN)

# ============================================================
# REQUEST MODEL
# ============================================================

class PredictionRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(
        ...,
        description="Exactly 24 chronological hourly records.",
        min_length=SEQ_LEN,
        max_length=SEQ_LEN,
    )

# ============================================================
# AQI CATEGORY — US AQI SCALE
# ============================================================

def get_aqi_category(aqi: float) -> str:
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    if aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    if aqi <= 200:
        return "Unhealthy"
    if aqi <= 300:
        return "Very Unhealthy"
    return "Hazardous"

# ============================================================
# PREPARE INPUT RECORDS
# ============================================================

def prepare_records(records: List[Dict[str, Any]]) -> np.ndarray:
    if len(records) != SEQ_LEN:
        raise HTTPException(
            status_code=400,
            detail=f"Exactly {SEQ_LEN} hourly records are required.",
        )

    matrix = []

    for record_index, record in enumerate(records):
        row = []

        for feature in FEATURES:
            if feature not in record:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Record {record_index} is missing "
                        f"required feature '{feature}'."
                    ),
                )

            try:
                value = float(record[feature])
            except (TypeError, ValueError, OverflowError):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Feature '{feature}' in record "
                        f"{record_index} must be numeric."
                    ),
                )

            if not np.isfinite(value):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Feature '{feature}' in record "
                        f"{record_index} must be finite."
                    ),
                )

            row.append(value)

        matrix.append(row)

    X_raw = np.asarray(matrix, dtype=np.float32)

    if X_raw.shape != (SEQ_LEN, NUM_FEATURES):
        raise HTTPException(
            status_code=400,
            detail="Input dimensions do not match the model configuration.",
        )

    return X_raw

# ============================================================
# PREPARE SCALED MODEL INPUT
# ============================================================

def scale_records(X_raw: np.ndarray) -> np.ndarray:
    try:
        X_scaled = scaler.transform(X_raw)
        X_scaled = np.asarray(X_scaled, dtype=np.float32)
    except Exception:
        logger.exception("Feature scaling failed.")
        raise HTTPException(
            status_code=500,
            detail="Feature scaling failed. Check the deployed scaler.",
        )

    if X_scaled.shape != (SEQ_LEN, NUM_FEATURES):
        raise HTTPException(
            status_code=500,
            detail="Scaled input dimensions do not match the model.",
        )

    if not np.isfinite(X_scaled).all():
        raise HTTPException(
            status_code=500,
            detail="Scaled input contains invalid numeric values.",
        )

    return X_scaled

# ============================================================
# RUN HYBRID PREDICTION
# ============================================================

def run_prediction(X_scaled: np.ndarray) -> Dict[str, float]:
    # LSTM input shape: (1, 24, number_of_features)
    X_lstm = X_scaled.reshape(1, SEQ_LEN, NUM_FEATURES)

    try:
        lstm_output = lstm_model(X_lstm, training=False)
        lstm_array = np.asarray(lstm_output).reshape(-1)

        if lstm_array.size == 0:
            raise ValueError("LSTM returned an empty prediction.")

        lstm_prediction = float(lstm_array[0])

    except Exception:
        logger.exception("LSTM inference failed.")
        raise HTTPException(
            status_code=500,
            detail="LSTM prediction failed. Check Render logs.",
        )

    # XGBoost input: latest hourly feature row.
    X_latest = X_scaled[-1].reshape(1, NUM_FEATURES)

    try:
        xgb_prediction = float(xgb_model.predict(X_latest)[0])
    except Exception:
        logger.exception("XGBoost inference failed.")
        raise HTTPException(
            status_code=500,
            detail="XGBoost prediction failed. Check Render logs.",
        )

    # Meta-model combines the two component predictions.
    meta_input = np.asarray(
        [[lstm_prediction, xgb_prediction]],
        dtype=np.float32,
    )

    try:
        hybrid_prediction = float(meta_model.predict(meta_input)[0])
    except Exception:
        logger.exception("Hybrid meta-model inference failed.")
        raise HTTPException(
            status_code=500,
            detail="Hybrid prediction failed. Check Render logs.",
        )

    predictions = [lstm_prediction, xgb_prediction, hybrid_prediction]

    if not all(np.isfinite(value) for value in predictions):
        raise HTTPException(
            status_code=500,
            detail="The model returned a non-finite prediction.",
        )

    return {
        "lstm_prediction": lstm_prediction,
        "xgboost_prediction": xgb_prediction,
        "hybrid_prediction": hybrid_prediction,
    }

# ============================================================
# SHAP IMPORT AND EXPLAINER
# ============================================================

@lru_cache(maxsize=1)
def get_shap_explainer():
    """
    Create and cache a SHAP TreeExplainer for the XGBoost model.
    This explains XGBoost, not the entire LSTM/meta-model pipeline.
    """
    try:
        import shap

        explainer = shap.TreeExplainer(xgb_model)
        return shap, explainer

    except Exception:
        logger.exception("Could not initialize the SHAP explainer.")
        raise HTTPException(
            status_code=503,
            detail=(
                "SHAP explainer could not be initialized. "
                "Check the installed SHAP/XGBoost versions and Render logs."
            ),
        )

# ============================================================
# SHAP HEALTH PROBE
# ============================================================

def check_shap_import():
    try:
        import shap
        return {
            "available": True,
            "version": getattr(shap, "__version__", "unknown"),
        }
    except Exception as exc:
        logger.warning("SHAP import check failed: %s", str(exc))
        return {
            "available": False,
            "version": None,
        }

# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    shap_status = check_shap_import()

    return {
        "status": "online",
        "service": "BreatheHealthy AI API",
        "version": app.version,
        "models": {
            "xgboost": xgb_model is not None,
            "lstm": lstm_model is not None,
            "stacked_hybrid": meta_model is not None,
            "shap": shap_status["available"],
        },
        "shap_version": shap_status["version"],
        "features": NUM_FEATURES,
        "sequence_length": SEQ_LEN,
        "prediction_target": "Next-hour US AQI",
        "explanation_target": "XGBoost component only",
    }

# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(request: PredictionRequest):
    X_raw = prepare_records(request.records)
    X_scaled = scale_records(X_raw)

    predictions = run_prediction(X_scaled)
    hybrid_prediction = predictions["hybrid_prediction"]

    return {
        "success": True,
        "prediction_target": "Next-hour US AQI",
        "predicted_aqi": round(hybrid_prediction, 2),
        "lstm_prediction": round(predictions["lstm_prediction"], 2),
        "xgboost_prediction": round(predictions["xgboost_prediction"], 2),
        "category": get_aqi_category(hybrid_prediction),
        "input_hours": SEQ_LEN,
        "features": NUM_FEATURES,
    }

# ============================================================
# SHAP EXPLANATION ENDPOINT
# ============================================================

@app.post("/explain")
def explain(request: PredictionRequest):
    X_raw = prepare_records(request.records)
    X_scaled = scale_records(X_raw)

    # Explain the same latest-hour scaled row used by XGBoost inference.
    X_latest = X_scaled[-1].reshape(1, NUM_FEATURES)

    shap, explainer = get_shap_explainer()

    try:
        explanation = explainer(X_latest)
        shap_values = np.asarray(explanation.values)

        # Typical regression output shape: (1, number_of_features).
        # Handle an optional trailing output dimension if present.
        if shap_values.ndim == 3 and shap_values.shape[0] == 1:
            shap_values = shap_values[0]

        if shap_values.ndim == 2:
            shap_row = shap_values[0]
        elif shap_values.ndim == 1:
            shap_row = shap_values
        else:
            raise ValueError(
                f"Unexpected SHAP output shape: {shap_values.shape}"
            )

        base_value_array = np.asarray(explanation.base_values).reshape(-1)

        if base_value_array.size == 0:
            raise ValueError("SHAP returned no base value.")

        base_value = float(base_value_array[0])

        if len(shap_row) != NUM_FEATURES:
            raise ValueError(
                f"Expected {NUM_FEATURES} SHAP values, got {len(shap_row)}."
            )

        if not np.isfinite(shap_row).all() or not np.isfinite(base_value):
            raise ValueError("SHAP returned non-finite values.")

    except Exception:
        logger.exception("SHAP calculation failed.")
        raise HTTPException(
            status_code=500,
            detail=(
                "SHAP calculation failed. Check Render logs for the "
                "underlying error and verify SHAP/XGBoost compatibility."
            ),
        )

    factors = []

    for index, feature in enumerate(FEATURES):
        contribution = float(shap_row[index])

        factors.append({
            "feature": feature,
            # Original unscaled value for user-facing context.
            "value": round(float(X_raw[-1, index]), 4),
            # Contribution is calculated against the scaled XGBoost input.
            "shap_value": round(contribution, 6),
            "impact": (
                "increases"
                if contribution > 0
                else "decreases"
                if contribution < 0
                else "neutral"
            ),
        })

    factors.sort(
        key=lambda item: abs(item["shap_value"]),
        reverse=True,
    )

    return {
        "success": True,
        "explanation_target": "XGBoost component prediction",
        "explanation_method": "SHAP TreeExplainer",
        "base_value": round(base_value, 6),
        "top_factors": factors[:10],
        "total_features": NUM_FEATURES,
        "note": (
            "SHAP values explain the XGBoost component only, not the "
            "complete LSTM plus meta-model hybrid prediction. Positive "
            "values push the XGBoost output upward; negative values push "
            "it downward."
        ),
    }

# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    return {
        "service": "BreatheHealthy AI API",
        "status": "online",
        "message": "Hybrid LSTM + XGBoost AQI forecasting service",
        "docs": "/docs",
        "health": "/health",
    }
