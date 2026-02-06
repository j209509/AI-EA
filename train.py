from __future__ import annotations
import argparse
import json
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix
import joblib

def compute_atr14(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    high = df["High"].astype(float).values
    low = df["Low"].astype(float).values
    close = df["Close"].astype(float).values
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    return pd.Series(tr).rolling(period).mean().values

def make_label_move(df: pd.DataFrame, bars_ahead: int = 3, thr: float = 15.0) -> np.ndarray:
    close = df["Close"].astype(float).values
    high = df["High"].astype(float).values
    low = df["Low"].astype(float).values
    y = np.full(len(df), np.nan)
    for i in range(len(df)-bars_ahead):
        maxh = np.max(high[i+1:i+1+bars_ahead])
        minl = np.min(low[i+1:i+1+bars_ahead])
        up = maxh - close[i]
        dn = close[i] - minl
        y[i] = 1.0 if (up >= thr or dn >= thr) else 0.0
    return y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="MT4 exported CSV (no header): Date,Time,Open,High,Low,Close,Volume")
    ap.add_argument("--out_model", default="model.joblib")
    ap.add_argument("--out_report", default="training_report.json")
    ap.add_argument("--thr", type=float, default=15.0)
    ap.add_argument("--bars_ahead", type=int, default=3)
    ap.add_argument("--atr_period", type=int, default=14)
    ap.add_argument("--train_ratio", type=float, default=0.7)
    args = ap.parse_args()

    df = pd.read_csv(args.csv, header=None)
    df.columns = ["Date","Time","Open","High","Low","Close","Volume"]

    df["ATR14"] = compute_atr14(df, period=args.atr_period)
    df["label"] = make_label_move(df, bars_ahead=args.bars_ahead, thr=args.thr)

    data = df.dropna(subset=["ATR14","label"]).copy()
    X = data[["ATR14"]].astype(float).values
    y = data["label"].astype(int).values

    n = len(y)
    split = int(n * args.train_ratio)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200, solver="lbfgs"))
    ])
    pipe.fit(X_train, y_train)

    p_test = pipe.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, p_test))

    def metrics_at(t: float):
        pred = (p_test >= t).astype(int)
        return {
            "threshold": t,
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
            "positive_rate_pred": float(pred.mean()),
            "positive_rate_true": float(y_test.mean()),
        }

    report = {
        "dataset_rows_total": int(len(df)),
        "dataset_rows_used": int(n),
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "label_definition": f"label=1 if within next {args.bars_ahead} bars either (max_high - close_now) >= {args.thr} OR (close_now - min_low) >= {args.thr}, else 0",
        "atr_period": int(args.atr_period),
        "threshold_move": float(args.thr),
        "roc_auc": auc,
        "metrics": [metrics_at(t) for t in [0.5, 0.6, 0.7]],
    }

    joblib.dump(pipe, args.out_model)
    with open(args.out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
