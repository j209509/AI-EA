from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()

model = joblib.load("model.joblib")

THRESHOLD = 0.52  # relaxed for EA usage

class PredictReq(BaseModel):
    atr14: float

class PredictRes(BaseModel):
    p_move: float
    decision: int
    decision_threshold: float

@app.post("/predict", response_model=PredictRes)
def predict(req: PredictReq):
    X = np.array([[req.atr14]], dtype=float)
    p = float(model.predict_proba(X)[0,1])
    d = 1 if p >= THRESHOLD else 0
    return PredictRes(p_move=p, decision=d, decision_threshold=THRESHOLD)

class DirRes(BaseModel):
    direction: str

@app.post("/direction", response_model=DirRes)
def direction():
    # simple placeholder; replace later with trained direction model
    return DirRes(direction="BUY")
