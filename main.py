
import os

# Limit CPU thread usage on the free Render instance.
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
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import tensorflow as tf
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# -------------------------------------------------------------------
# Application configuration
# -------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("breathehealthy")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

SEQUENCE_LENGTH = 24
EXPECTED_FEATURE_COUNT = 33
SHAP_TOP_COUNT = 10

app = FastAPI(
    title="BreatheHealthy API",
    description="Hybrid air-quality forecasting and explainability API",
    version="1.0.0",
)

# Allow frontend requests. Restrict origins to your frontend domain
# if you later need stricter production CORS settings.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://breathe-wise-predict.base44.app",
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Load the already-trained models. Do not retrain them here.
# -------------------------------------------------------------------

logger.info("Loading BreatheHealthy models...")

try:
    lstm_model = tf.keras.models.load_model(
        MODEL_DIR / "lstm_model_final.keras",
        compile=False,
    )

    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(str(MODEL_DIR / "xgb_model.json"))

    scaler = joblib.load(MODEL_DIR / "scaler_final.pkl")
    meta_model = joblib.load(MODEL_DIR / "meta_model_final.pkl")

    with open(MODEL_DIR / "features_final.json", "r", encoding="utf-8") as f:
        features = json.load(f)

    if isinstance(features, dict):
        # Supports JSON saved as {"features": [...]}.
        features = features.get("features", features.get("feature_names"))

    if not isinstance(features, list):
        raise ValueError(
            "features_final.json must contain a list of feature names."
        )

    features = [str(name) for name in features]

    if len(features) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} features, "
            f"but features_final.json contains {len(features)}."
        )

    if len(set(features)) != len(features):
        raise ValueError("features_final.json contains duplicate feature names.")

    logger.info(
        "Models loaded. Features=%d, sequence=%d",
        len(features),
        SEQUENCE_LENGTH,
    )

except Exception:
    logger.exception("Failed to load one or more model files.")
    raise

# Only one SHAP request may run at a time on this small instance.
shap_lock = threading.Lock()
shap_explainer = None

# -------------------------------------------------------------------
# Request models
# -------------------------------------------------------------------

class PredictionRequest(BaseModel):
    records: list[dict[str, Any]] = Field(
        ...,
        min_length=SEQUENCE_LENGTH,
        max_length=SEQUENCE_LENGTH,
        description="Exactly 24 chronological records containing model features",
    )


class ExplainRequest(BaseModel):
    records: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        max_length=SEQUENCE_LENGTH,
        description="One to 24 records containing model features",
    )


# -------------------------------------------------------------------
# Data preparation
# -------------------------------------------------------------------

def prepare_row(record: dict[str, Any]) -> np.ndarray:
    """Convert one record into the exact feature order used for training."""
    values = []

    for feature in features:
        if feature not in record:
            raise HTTPException(
                status_code=422,
                detail=f"Missing required feature: {feature}",
            )

        try:
            value = float(record[feature])
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=422,
                detail=f"Feature '{feature}' must be numeric.",
            )

        if not np.isfinite(value):
            raise HTTPException(
                status_code=422,
                detail=f"Feature '{feature}' must be a finite number.",
            )

        values.append(value)

    return np.asarray(values, dtype=np.float32)


def prepare_records(records: list[dict[str, Any]]) -> np.ndarray:
    """Convert records to a 2-D array in the trained feature order."""
    if not records:
        raise HTTPException(status_code=422, detail="No records were provided.")

    rows = [prepare_row(record) for record in records]
    matrix = np.stack(rows, axis=0)

    if matrix.shape[1] != EXPECTED_FEATURE_COUNT:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {EXPECTED_FEATURE_COUNT} features per record.",
        )

    return matrix


def scale_matrix(matrix: np.ndarray) -> np.ndarray:
    """Apply the existing fitted scaler without fitting or retraining."""
    try:
        scaled = scaler.transform(matrix)
    except Exception as exc:
        logger.exception("Feature scaling failed.")
        raise HTTPException(
            status_code=422,
            detail=f"Feature scaling failed: {type(exc).__name__}.",
        )

    scaled = np.asarray(scaled, dtype=np.float32)

    if not np.isfinite(scaled).all():
        raise HTTPException(
            status_code=422,
            detail="Scaled feature values contain invalid numbers.",
        )

    return scaled


# -------------------------------------------------------------------
# Hybrid prediction
# -------------------------------------------------------------------

def run_prediction(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Run the existing LSTM + XGBoost + meta-model pipeline."""
    raw_matrix = prepare_records(records)

    if raw_matrix.shape[0] != SEQUENCE_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=f"Prediction requires exactly {SEQUENCE_LENGTH} records.",
        )

    scaled_matrix = scale_matrix(raw_matrix)

    # LSTM input shape: (batch, sequence length, feature count).
    lstm_input = scaled_matrix.reshape(
        1,
        SEQUENCE_LENGTH,
        EXPECTED_FEATURE_COUNT,
    )

    try:
        lstm_output = np.asarray(
            lstm_model.predict(lstm_input, verbose=0)
        ).reshape(-1)

        # XGBoost uses the most recent time step.
        xgb_input = scaled_matrix[-1:, :]
        xgb_output = np.asarray(
            xgb_model.predict(xgb_input)
        ).reshape(-1)

        if lstm_output.size == 0 or xgb_output.size == 0:
            raise ValueError("A model returned an empty prediction.")

        lstm_prediction = float(lstm_output[0])
        xgb_prediction = float(xgb_output[0])

        # Meta-model combines the two base-model predictions.
        meta_input = np.asarray(
            [[lstm_prediction, xgb_prediction]],
            dtype=np.float32,
        )

        final_output = np.asarray(
            meta_model.predict(meta_input)
        ).reshape(-1)

        if final_output.size == 0:
            raise ValueError("The meta-model returned an empty prediction.")

        final_prediction = float(final_output[0])

        if not np.isfinite(
            [lstm_prediction, xgb_prediction, final_prediction]
        ).all():
            raise ValueError("A model returned a non-finite prediction.")

        return {
            "prediction": final_prediction,
            "predicted_aqi": final_prediction,
            "lstm_prediction": lstm_prediction,
            "xgb_prediction": xgb_prediction,
            "model": "LSTM + XGBoost + Meta-Learner",
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed.")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {type(exc).__name__}.",
        )
    finally:
        gc.collect()


# -------------------------------------------------------------------
# Health check
# -------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "BreatheHealthy API",
        "status": "online",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    # Do not import SHAP here: health checks should remain lightweight.
    return {
        "status": "healthy",
        "models": {
            "lstm": "loaded",
            "xgboost": "loaded",
            "scaler": "loaded",
            "meta_model": "loaded",
            "shap": "loaded on demand",
        },
        "features": len(features),
        "sequence_length": SEQUENCE_LENGTH,
    }


# -------------------------------------------------------------------
# Prediction endpoint
# -------------------------------------------------------------------

@app.post("/predict")
def predict(request: PredictionRequest):
    return run_prediction(request.records)


# -------------------------------------------------------------------
# SHAP explanation endpoint
# -------------------------------------------------------------------

@app.post("/explain")
def explain(request: ExplainRequest):
    global shap_explainer

    # Never queue multiple memory-heavy SHAP requests on the free instance.
    if not shap_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=503,
            detail="SHAP explanation is busy. Please try again shortly.",
        )

    explainer = None
    shap_values = None
    explanation = None
    raw_matrix = None
    scaled_matrix = None
    xgb_input = None
    values = None
    base_value = None

    try:
        # The frontend may send just the latest record.
        raw_matrix = prepare_records(request.records)
        scaled_matrix = scale_matrix(raw_matrix)
        xgb_input = scaled_matrix[-1:, :]

        # Import SHAP only when an explanation is requested.
        import shap

        if shap_explainer is None:
            logger.info("Creating SHAP TreeExplainer on demand.")
            shap_explainer = shap.TreeExplainer(xgb_model)

        explainer = shap_explainer
        explanation = explainer(xgb_input)

        values = np.asarray(explanation.values)

        # Normalize common SHAP output shapes for one-row regression.
        if values.ndim == 3:
            values = values[0, :, 0]
        elif values.ndim == 2:
            values = values[0]

        values = np.asarray(values).reshape(-1)

        if values.size != len(features):
            raise ValueError(
                f"SHAP returned {values.size} values for "
                f"{len(features)} features."
            )

        base_value = np.asarray(explanation.base_values).reshape(-1)
        base = float(base_value[0]) if base_value.size else 0.0

        if not np.isfinite(values).all() or not np.isfinite(base):
            raise ValueError("SHAP returned non-finite values.")

        # Report original feature values for readability.
        latest_record = request.records[-1]

        factor_items = []
        for index, feature in enumerate(features):
            shap_value = float(values[index])
            raw_value = float(raw_matrix[-1, index])

            factor_items.append(
                {
                    "feature": feature,
                    "value": raw_value,
                    "shap_value": shap_value,
                    "impact": (
                        "increases prediction"
                        if shap_value > 0
                        else "decreases prediction"
                        if shap_value < 0
                        else "neutral"
                    ),
                }
            )

        factor_items.sort(
            key=lambda item: abs(item["shap_value"]),
            reverse=True,
        )

        return {
            "status": "success",
            "model": "XGBoost component",
            "base_value": base,
            "top_factors": factor_items[:SHAP_TOP_COUNT],
            "total_features": len(features),
        }

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "SHAP calculation failed: %s: %s",
            type(exc).__name__,
            str(exc),
        )
        raise HTTPException(
            status_code=503,
            detail=(
                f"SHAP explanation failed ({type(exc).__name__}). "
                "The AQI prediction is still available."
            ),
        )

    finally:
        # Do not clear the prediction or modify any trained model.
        del explainer
        del shap_values
        del explanation
        del raw_matrix
        del scaled_matrix
        del xgb_input
        del values
        del base_value
        gc.collect()
        shap_lock.release()
