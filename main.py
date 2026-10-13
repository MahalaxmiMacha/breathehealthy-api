
import os

# Set limits before numerical/ML imports.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import gc
import json
import logging
import threading
from typing import Any, Dict, List

import joblib
import numpy as np
import tensorflow as tf
import xgboost as xgb

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("breathehealthy")

try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    logger.warning("Could not set TensorFlow thread limits.")

app = FastAPI(
    title="BreatheHealthy AI API",
    description="Hybrid LSTM + XGBoost Air Quality Forecasting API",
    version="1.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

LSTM_PATH = os.path.join(MODEL_DIR, "lstm_model_final.keras")
XGB_PATH = os.path.join(MODEL_DIR, "xgb_model.json")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_final.pkl")
META_PATH = os.path.join(MODEL_DIR, "meta_model_final.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "features_final.json")

SEQ_LEN = 24

# Only one SHAP calculation at a time.
shap_lock = threading.Lock()

logger.info("Loading BreatheHealthy models...")

required_files = [
    LSTM_PATH,
    XGB_PATH,
    SCALER_PATH,
    META_PATH,
    FEATURES_PATH,
]

missing_files = [
    path for path in required_files if not os.path.isfile(path)
]

if missing_files:
    raise RuntimeError(
        "Missing model files: "
        + ", ".join(os.path.basename(p) for p in missing_files)
    )

# Load each trained model once.
lstm_model = tf.keras.models.load_model(
    LSTM_PATH, compile=False
)

xgb_model = xgb.XGBRegressor()
xgb_model.load_model(XGB_PATH)
xgb_model.set_params(n_jobs=1)

scaler = joblib.load(SCALER_PATH)
meta_model = joblib.load(META_PATH)

with open(FEATURES_PATH, "r", encoding="utf-8") as f:
    feature_data = json.load(f)

FEATURES = (
    feature_data.get("features", feature_data.get("FEATURES"))
    if isinstance(feature_data, dict)
    else feature_data
)

if not isinstance(FEATURES, list) or not FEATURES:
    raise RuntimeError("Invalid features_final.json.")

if (
    not all(isinstance(x, str) and x for x in FEATURES)
    or len(set(FEATURES)) != len(FEATURES)
):
    raise RuntimeError("Invalid or duplicate feature names.")

NUM_FEATURES = len(FEATURES)

scaler_count = getattr(scaler, "n_features_in_", None)
if scaler_count is not None and scaler_count != NUM_FEATURES:
    raise RuntimeError("Scaler and feature-list sizes do not match.")

logger.info("Models loaded. Features=%s, sequence=%s",
            NUM_FEATURES, SEQ_LEN)


class PredictionRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(
        ..., min_length=SEQ_LEN, max_length=SEQ_LEN
    )


# /explain accepts one row from Base44, or 24 rows from older clients.
class ExplainRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(
        ..., min_length=1, max_length=SEQ_LEN
    )


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


def prepare_row(record: Dict[str, Any], index: int = 0) -> np.ndarray:
    values = []

    for feature in FEATURES:
        if feature not in record:
            raise HTTPException(
                status_code=400,
                detail=f"Record {index} is missing '{feature}'.",
            )

        try:
            value = float(record[feature])
        except (TypeError, ValueError, OverflowError):
            raise HTTPException(
                status_code=400,
                detail=f"Feature '{feature}' must be numeric.",
            )

        if not np.isfinite(value):
            raise HTTPException(
                status_code=400,
                detail=f"Feature '{feature}' must be finite.",
            )

        values.append(value)

    return np.asarray([values], dtype=np.float32)


def prepare_records(records: List[Dict[str, Any]]) -> np.ndarray:
    if len(records) != SEQ_LEN:
        raise HTTPException(
            status_code=400,
            detail=f"Exactly {SEQ_LEN} hourly records are required.",
        )

    return np.concatenate(
        [prepare_row(record, i) for i, record in enumerate(records)],
        axis=0,
    )


def scale_matrix(X_raw: np.ndarray) -> np.ndarray:
    try:
        X_scaled = np.asarray(
            scaler.transform(X_raw), dtype=np.float32
        )
    except Exception:
        logger.exception("Feature scaling failed.")
        raise HTTPException(
            status_code=500, detail="Feature scaling failed."
        )

    if (
        X_scaled.shape != X_raw.shape
        or not np.isfinite(X_scaled).all()
    ):
        raise HTTPException(
            status_code=500,
            detail="Scaled input has invalid dimensions or values.",
        )

    return X_scaled


def run_prediction(X_scaled: np.ndarray) -> Dict[str, float]:
    X_lstm = X_scaled.reshape(1, SEQ_LEN, NUM_FEATURES)

    try:
        output = lstm_model(X_lstm, training=False)
        lstm_prediction = float(np.asarray(output).reshape(-1)[0])
    except Exception:
        logger.exception("LSTM inference failed.")
        raise HTTPException(status_code=500, detail="LSTM prediction failed.")

    X_latest = X_scaled[-1:].copy()

    try:
        xgb_prediction = float(xgb_model.predict(X_latest)[0])
    except Exception:
        logger.exception("XGBoost inference failed.")
        raise HTTPException(status_code=500, detail="XGBoost prediction failed.")

    try:
        meta_input = np.asarray(
            [[lstm_prediction, xgb_prediction]], dtype=np.float32
        )
        hybrid_prediction = float(meta_model.predict(meta_input)[0])
    except Exception:
        logger.exception("Hybrid inference failed.")
        raise HTTPException(status_code=500, detail="Hybrid prediction failed.")

    results = [lstm_prediction, xgb_prediction, hybrid_prediction]
    if not all(np.isfinite(value) for value in results):
        raise HTTPException(
            status_code=500, detail="Model returned invalid predictions."
        )

    return {
        "lstm_prediction": lstm_prediction,
        "xgboost_prediction": xgb_prediction,
        "hybrid_prediction": hybrid_prediction,
    }


@app.get("/")
def root():
    return {
        "service": "BreatheHealthy AI API",
        "status": "online",
        "message": "Hybrid LSTM + XGBoost AQI forecasting service",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    # IMPORTANT: Do not import SHAP here. Health checks can be frequent.
    return {
        "status": "online",
        "service": "BreatheHealthy AI API",
        "version": app.version,
        "models": {
            "xgboost": xgb_model is not None,
            "lstm": lstm_model is not None,
            "stacked_hybrid": meta_model is not None,
            "shap": "loaded on demand",
        },
        "features": NUM_FEATURES,
        "sequence_length": SEQ_LEN,
        "prediction_target": "Next-hour US AQI",
        "explanation_target": "XGBoost component only",
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    # Keep the original 24-hour LSTM input contract.
    X_raw = prepare_records(request.records)
    X_scaled = scale_matrix(X_raw)
    predictions = run_prediction(X_scaled)

    hybrid = predictions["hybrid_prediction"]

    return {
        "success": True,
        "prediction_target": "Next-hour US AQI",
        "predicted_aqi": round(hybrid, 2),
        "lstm_prediction": round(predictions["lstm_prediction"], 2),
        "xgboost_prediction": round(predictions["xgboost_prediction"], 2),
        "category": get_aqi_category(hybrid),
        "input_hours": SEQ_LEN,
        "features": NUM_FEATURES,
    }


@app.post("/explain")
def explain(request: ExplainRequest):
    # Do not run multiple memory-intensive explanations simultaneously.
    if not shap_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=503,
            detail="SHAP explanation is busy. Please retry shortly.",
        )

    X_raw = None
    X_scaled = None
    X_latest = None
    explanation = None
    explainer = None
    shap_module = None

    try:
        # Accept a single row from Base44 or use the last row of 24.
        latest_record = request.records[-1]
        X_raw = prepare_row(latest_record)
        X_scaled = scale_matrix(X_raw)
        X_latest = X_scaled.copy()

        # SHAP is imported only when someone requests an explanation.
        try:
            import shap as shap_module
        except Exception:
            logger.exception("SHAP import failed.")
            raise HTTPException(
                status_code=503,
                detail="SHAP is temporarily unavailable.",
            )

        # Reuse the already-loaded XGBoost model. No second model copy,
        # no background dataset, and only one explained observation.
        explainer = shap_module.TreeExplainer(xgb_model)
        explanation = explainer(X_latest)

        shap_values = np.asarray(explanation.values)
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

        base_values = np.asarray(explanation.base_values).reshape(-1)
        if base_values.size == 0:
            raise ValueError("SHAP returned no base value.")

        base_value = float(base_values[0])

        if (
            len(shap_row) != NUM_FEATURES
            or not np.isfinite(shap_row).all()
            or not np.isfinite(base_value)
        ):
            raise ValueError("SHAP returned invalid values.")

        factors = []
        for i, feature in enumerate(FEATURES):
            contribution = float(shap_row[i])
            factors.append({
                "feature": feature,
                "value": round(float(X_raw[0, i]), 4),
                "shap_value": round(contribution, 6),
                "impact": (
                    "increases" if contribution > 0
                    else "decreases" if contribution < 0
                    else "neutral"
                ),
            })

        factors.sort(
            key=lambda item: abs(item["shap_value"]), reverse=True
        )

        return {
            "success": True,
            "explanation_target": "XGBoost component prediction",
            "explanation_method": "SHAP TreeExplainer",
            "base_value": round(base_value, 6),
            "top_factors": factors[:10],
            "total_features": NUM_FEATURES,
            "note": (
                "SHAP explains the XGBoost component, not the complete "
                "LSTM plus meta-model hybrid prediction. Positive values "
                "push the XGBoost output upward; negative values push it "
                "downward."
            ),
        }

    except HTTPException:
        raise
    except Exception:
        logger.exception("SHAP calculation failed.")
        raise HTTPException(
            status_code=503,
            detail="SHAP explanation failed. The AQI prediction is separate.",
        )
    finally:
        # Release request-specific explanation objects.
        del explanation, explainer, shap_module
        del X_raw, X_scaled, X_latest
        gc.collect()
        shap_lock.release()
