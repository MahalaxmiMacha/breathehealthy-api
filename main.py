
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
    version="1.1.0",
)

# -------------------------------------------------------------------
# CORS configuration
# -------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://breathe-wise-predict.base44.app",
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_origin_regex=r"^https://([a-zA-Z0-9-]+\.)*base44\.app$",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Load existing trained models. No training occurs here.
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

    with open(
        MODEL_DIR / "features_final.json",
        "r",
        encoding="utf-8",
    ) as f:
        features = json.load(f)

    if isinstance(features, dict):
        features = features.get(
            "features",
            features.get("feature_names"),
        )

    if not isinstance(features, list):
        raise ValueError(
            "features_final.json must contain a list of feature names."
        )

    features = [str(name) for name in features]

    if len(features) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} features, "
            f"but found {len(features)}."
        )

    if len(set(features)) != len(features):
        raise ValueError(
            "features_final.json contains duplicate feature names."
        )

    logger.info(
        "Models loaded. Features=%d, sequence=%d",
        len(features),
        SEQUENCE_LENGTH,
    )

except Exception:
    logger.exception("Failed to load model files.")
    raise

# Prevent simultaneous explanation calculations on the small instance.
explain_lock = threading.Lock()

# -------------------------------------------------------------------
# Request models
# -------------------------------------------------------------------

class PredictionRequest(BaseModel):
    records: list[dict[str, Any]] = Field(
        ...,
        min_length=SEQUENCE_LENGTH,
        max_length=SEQUENCE_LENGTH,
        description="Exactly 24 chronological records",
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
    """Read features in exactly the order used during training."""
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
                detail=f"Feature '{feature}' must be finite.",
            )

        values.append(value)

    return np.asarray(values, dtype=np.float32)


def prepare_records(records: list[dict[str, Any]]) -> np.ndarray:
    """Convert records into a 2-D array in training feature order."""
    if not records:
        raise HTTPException(
            status_code=422,
            detail="No records were provided.",
        )

    matrix = np.stack(
        [prepare_row(record) for record in records],
        axis=0,
    )

    if matrix.shape[1] != EXPECTED_FEATURE_COUNT:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {EXPECTED_FEATURE_COUNT} features.",
        )

    return matrix


def scale_matrix(matrix: np.ndarray) -> np.ndarray:
    """Use the saved scaler; never fit or retrain it."""
    try:
        scaled = scaler.transform(matrix)
    except Exception as exc:
        logger.exception("Feature scaling failed.")
        raise HTTPException(
            status_code=422,
            detail=f"Feature scaling failed: {type(exc).__name__}.",
        ) from exc

    scaled = np.asarray(scaled, dtype=np.float32)

    if not np.isfinite(scaled).all():
        raise HTTPException(
            status_code=422,
            detail="Scaled features contain invalid numbers.",
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

    try:
        lstm_input = scaled_matrix.reshape(
            1,
            SEQUENCE_LENGTH,
            EXPECTED_FEATURE_COUNT,
        )

        lstm_output = np.asarray(
            lstm_model.predict(lstm_input, verbose=0)
        ).reshape(-1)

        # XGBoost uses the latest time step.
        xgb_input = scaled_matrix[-1:, :]
        xgb_output = np.asarray(
            xgb_model.predict(xgb_input)
        ).reshape(-1)

        if lstm_output.size == 0 or xgb_output.size == 0:
            raise ValueError("A base model returned no prediction.")

        lstm_prediction = float(lstm_output[0])
        xgb_prediction = float(xgb_output[0])

        # Preserve the original meta-model input order.
        meta_input = np.asarray(
            [[lstm_prediction, xgb_prediction]],
            dtype=np.float32,
        )

        final_output = np.asarray(
            meta_model.predict(meta_input)
        ).reshape(-1)

        if final_output.size == 0:
            raise ValueError("The meta-model returned no prediction.")

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
        ) from exc
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
    return {
        "status": "healthy",
        "models": {
            "lstm": "loaded",
            "xgboost": "loaded",
            "scaler": "loaded",
            "meta_model": "loaded",
            "explanation": "native XGBoost contributions",
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
# Explanation endpoint
# Uses XGBoost native feature contributions instead of shap.TreeExplainer.
# No standalone SHAP import or explainer initialization is required.
# -------------------------------------------------------------------

@app.post("/explain")
def explain(request: ExplainRequest):
    if not explain_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=503,
            detail="An explanation is already running. Please try again.",
        )

    raw_matrix = None
    scaled_matrix = None
    dmatrix = None
    contributions = None

    try:
        raw_matrix = prepare_records(request.records)
        scaled_matrix = scale_matrix(raw_matrix)

        # Explain the latest record, using the same scaled features
        # supplied to the XGBoost component during prediction.
        latest_scaled_row = scaled_matrix[-1:, :]

        # Native XGBoost TreeSHAP contributions. The last element
        # is the bias/base contribution; preceding elements map to
        # the trained input features.
        dmatrix = xgb.DMatrix(latest_scaled_row)

        contributions = xgb_model.get_booster().predict(
            dmatrix,
            pred_contribs=True,
        )

        contributions = np.asarray(contributions)

        # For a single-row regression model, expected shape is
        # (1, feature_count + 1).
        if contributions.ndim != 2 or contributions.shape[0] != 1:
            raise ValueError(
                f"Unexpected contribution shape: {contributions.shape}"
            )

        row_contributions = contributions[0]

        if row_contributions.size != len(features) + 1:
            raise ValueError(
                "XGBoost contribution count does not match "
                "the feature count plus the bias term."
            )

        feature_contributions = row_contributions[:-1]
        base_value = float(row_contributions[-1])

        if (
            not np.isfinite(feature_contributions).all()
            or not np.isfinite(base_value)
        ):
            raise ValueError("XGBoost returned non-finite contributions.")

        factors = []

        for index, feature in enumerate(features):
            contribution = float(feature_contributions[index])

            if contribution > 1e-8:
                impact = "increases"
            elif contribution < -1e-8:
                impact = "decreases"
            else:
                impact = "neutral"

            factors.append({
                "feature": feature,
                "value": float(raw_matrix[-1, index]),
                "shap_value": contribution,
                "impact": impact,
            })

        factors.sort(
            key=lambda item: abs(item["shap_value"]),
            reverse=True,
        )

        logger.info(
            "Explanation calculated successfully for %d features.",
            len(features),
        )

        return {
            "status": "success",
            "model": "XGBoost component",
            "explanation_method": "XGBoost native TreeSHAP contributions",
            "base_value": base_value,
            "top_factors": factors[:SHAP_TOP_COUNT],
            "total_features": len(features),
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Explanation calculation failed.")
        raise HTTPException(
            status_code=503,
            detail=(
                f"Explanation failed ({type(exc).__name__}). "
                "The AQI prediction endpoint is separate."
            ),
        ) from exc

    finally:
        raw_matrix = None
        scaled_matrix = None
        dmatrix = None
        contributions = None
        gc.collect()
        explain_lock.release()
