import os

# ============================================================
# RENDER / LOW-MEMORY OPTIMIZATION
# These MUST be set before importing TensorFlow / XGBoost.
# ============================================================

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import joblib
import numpy as np
import tensorflow as tf
import xgboost as xgb

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any


# ============================================================
# TENSORFLOW THREAD LIMIT
# ============================================================

try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    pass


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="BreatheHealthy AI API",
    description="Hybrid LSTM + XGBoost Air Quality Forecasting API",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

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

print("Loading BreatheHealthy models...")

LSTM_PATH = os.path.join(
    MODEL_DIR,
    "lstm_model_final.keras"
)

XGB_PATH = os.path.join(
    MODEL_DIR,
    "xgb_model.json"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "scaler_final.pkl"
)

META_PATH = os.path.join(
    MODEL_DIR,
    "meta_model_final.pkl"
)

FEATURES_PATH = os.path.join(
    MODEL_DIR,
    "features_final.json"
)


# ============================================================
# LOAD TRAINED MODELS
# ============================================================

# LSTM
lstm_model = tf.keras.models.load_model(
    LSTM_PATH,
    compile=False
)

# XGBoost
xgb_model = xgb.XGBRegressor()
xgb_model.load_model(XGB_PATH)

# Important for low-memory Render environment
try:
    xgb_model.set_params(n_jobs=1)
except Exception:
    pass

# Scaler
scaler = joblib.load(SCALER_PATH)

# Hybrid meta-model
meta_model = joblib.load(META_PATH)


# ============================================================
# LOAD FEATURES
# ============================================================

with open(FEATURES_PATH, "r") as f:
    features_data = json.load(f)

if isinstance(features_data, dict):
    FEATURES = features_data.get(
        "features",
        features_data.get("FEATURES")
    )
else:
    FEATURES = features_data

if not FEATURES:
    raise RuntimeError(
        "Could not load feature list from features_final.json"
    )


# ============================================================
# MODEL CONFIGURATION
# ============================================================

SEQ_LEN = 24
NUM_FEATURES = len(FEATURES)

print("Models loaded successfully.")
print("Features:", NUM_FEATURES)
print("Sequence length:", SEQ_LEN)


# ============================================================
# REQUEST MODEL
# ============================================================

class PredictionRequest(BaseModel):
    records: List[Dict[str, Any]]


# ============================================================
# AQI CATEGORY
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
# PREPARE INPUT RECORDS
# ============================================================

def prepare_records(records):

    # Must have exactly 24 hourly records
    if len(records) != SEQ_LEN:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Exactly {SEQ_LEN} consecutive "
                f"hourly records are required."
            )
        )

    try:

        matrix = []

        for record in records:

            row = []

            for feature in FEATURES:

                # Check feature exists
                if feature not in record:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Missing feature: {feature}"
                    )

                # Convert to float
                value = float(record[feature])

                # Check for NaN / infinity
                if not np.isfinite(value):
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Invalid value for feature: "
                            f"{feature}"
                        )
                    )

                row.append(value)

            matrix.append(row)

        # Float32 reduces memory usage
        X_raw = np.asarray(
            matrix,
            dtype=np.float32
        )

        return X_raw

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input data: {str(e)}"
        )


# ============================================================
# HEALTH CHECK
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

        "features": NUM_FEATURES,

        "sequence_length": SEQ_LEN,

        "prediction_target": "Next-hour US AQI"
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(request: PredictionRequest):

    # --------------------------------------------------------
    # 1. Prepare input
    # --------------------------------------------------------

    X_raw = prepare_records(request.records)


    # --------------------------------------------------------
    # 2. Scale using the SAME scaler used during training
    # --------------------------------------------------------

    try:
        X_scaled = scaler.transform(X_raw)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Scaling failed: {str(e)}"
        )


    # --------------------------------------------------------
    # 3. Prepare LSTM sequence
    # Shape:
    # (1, 24, 33)
    # --------------------------------------------------------

    X_lstm = X_scaled.reshape(
        1,
        SEQ_LEN,
        NUM_FEATURES
    )


    # --------------------------------------------------------
    # 4. LSTM prediction
    #
    # Direct model call is lighter than model.predict()
    # --------------------------------------------------------

    try:

        lstm_output = lstm_model(
            X_lstm,
            training=False
        )

        lstm_prediction = float(
            lstm_output.numpy()[0][0]
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"LSTM prediction failed: {str(e)}"
        )


    # --------------------------------------------------------
    # 5. XGBoost prediction
    #
    # The hybrid model uses the latest hour's features
    # --------------------------------------------------------

    try:

        X_xgb = X_scaled[-1].reshape(
            1,
            NUM_FEATURES
        )

        xgb_prediction = float(
            xgb_model.predict(
                X_xgb
            )[0]
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"XGBoost prediction failed: {str(e)}"
        )


    # --------------------------------------------------------
    # 6. Hybrid meta-model
    #
    # Inputs:
    # [LSTM prediction, XGBoost prediction]
    # --------------------------------------------------------

    try:

        meta_input = np.asarray(
            [[
                lstm_prediction,
                xgb_prediction
            ]],
            dtype=np.float32
        )

        hybrid_prediction = float(
            meta_model.predict(
                meta_input
            )[0]
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Hybrid prediction failed: {str(e)}"
        )


    # --------------------------------------------------------
    # 7. AQI category
    # --------------------------------------------------------

    category = get_aqi_category(
        hybrid_prediction
    )


    # --------------------------------------------------------
    # 8. Response
    # --------------------------------------------------------

    return {

        "success": True,

        "prediction_target":
            "Next-hour US AQI",

        "predicted_aqi":
            round(hybrid_prediction, 2),

        "lstm_prediction":
            round(lstm_prediction, 2),

        "xgboost_prediction":
            round(xgb_prediction, 2),

        "category":
            category,

        "input_hours":
            SEQ_LEN,

        "features":
            NUM_FEATURES
    }


# ============================================================
# SHAP EXPLANATION ENDPOINT
# ============================================================

@app.post("/explain")
def explain(request: PredictionRequest):

    # IMPORTANT:
    # SHAP is imported ONLY when this endpoint is called.
    # This keeps the normal prediction service lighter.
    import shap


    # --------------------------------------------------------
    # 1. Prepare input
    # --------------------------------------------------------

    X_raw = prepare_records(
        request.records
    )


    # --------------------------------------------------------
    # 2. Scale
    # --------------------------------------------------------

    try:

        X_scaled = scaler.transform(
            X_raw
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Scaling failed: {str(e)}"
        )


    # --------------------------------------------------------
    # 3. Use latest hour for XGBoost explanation
    # --------------------------------------------------------

    X_latest = X_scaled[-1].reshape(
        1,
        NUM_FEATURES
    )


    # --------------------------------------------------------
    # 4. Create SHAP TreeExplainer
    # --------------------------------------------------------

    try:

        explainer = shap.TreeExplainer(
            xgb_model
        )

        shap_values = explainer.shap_values(
            X_latest
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"SHAP explanation failed: {str(e)}"
        )


    # --------------------------------------------------------
    # 5. Handle SHAP output
    # --------------------------------------------------------

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_row = np.asarray(
        shap_values
    )[0]


    # --------------------------------------------------------
    # 6. Base value
    # --------------------------------------------------------

    base_value = explainer.expected_value

    if isinstance(base_value, np.ndarray):

        base_value = float(
            base_value.reshape(-1)[0]
        )

    else:

        base_value = float(
            base_value
        )


    # --------------------------------------------------------
    # 7. Build feature explanations
    # --------------------------------------------------------

    factors = []

    for i, feature in enumerate(FEATURES):

        shap_value = float(
            shap_row[i]
        )

        raw_value = float(
            X_raw[-1][i]
        )

        if shap_value > 0:
            impact = "increases"
        else:
            impact = "decreases"

        factors.append({

            "feature":
                feature,

            "value":
                round(raw_value, 4),

            "shap_value":
                round(shap_value, 4),

            "impact":
                impact
        })


    # --------------------------------------------------------
    # 8. Sort by absolute SHAP importance
    # --------------------------------------------------------

    factors.sort(
        key=lambda x: abs(
            x["shap_value"]
        ),
        reverse=True
    )


    # --------------------------------------------------------
    # 9. Return top 10 factors
    # --------------------------------------------------------

    return {

        "success": True,

        "explanation_target":
            "Next-hour AQI",

        "base_value":
            round(base_value, 3),

        "top_factors":
            factors[:10],

        "total_features":
            NUM_FEATURES
    }


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {

        "service":
            "BreatheHealthy AI API",

        "status":
            "online",

        "message":
            "Hybrid LSTM + XGBoost AQI forecasting service",

        "docs":
            "/docs",

        "health":
            "/health"
    }
