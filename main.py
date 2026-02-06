from __future__ import annotations
import os
import json
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np

APP_NAME = "xau-m5-move3-thr15-atr14"
MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.6"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")

app = FastAPI(title=APP_NAME)

class PredictRequest(BaseModel):
    atr14: float = Field(..., description="ATR(14) value at bar close")

class PredictResponse(BaseModel):
    p_move: float
    decision: int
    decision_threshold: float
    model_version: str

def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")

MODEL = load_model()

@app.get("/health")
def health():
    return {
        "ok": True,
        "app": APP_NAME,
        "model_version": MODEL_VERSION,
        "decision_threshold": DECISION_THRESHOLD,
    }

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        x = np.array([[float(req.atr14)]], dtype=float)
        p = float(MODEL.predict_proba(x)[0, 1])
        decision = 1 if p >= DECISION_THRESHOLD else 0
        return PredictResponse(
            p_move=p,
            decision=decision,
            decision_threshold=DECISION_THRESHOLD,
            model_version=MODEL_VERSION,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
