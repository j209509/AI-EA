このAPIは、XAUUSDのM5を想定し、ATR14から「3本先で±15以上動くか（どちら方向でも可）」を二値分類します。

ラベル定義
・label=1: 次の3本のどこかで、(max_high - close_now) >= 15 または (close_now - min_low) >= 15
・label=0: 上記を満たさない

学習
python train.py --csv XAUUSDm50206.csv --thr 15 --bars_ahead 3

起動（ローカル）
uvicorn main:app --reload --port 8080

Cloud Run
Dockerfileをそのまま使えます。
環境変数:
・DECISION_THRESHOLD（例: 0.6）
・MODEL_VERSION（任意）
・MODEL_PATH（通常は model.joblib のままでOK）

API
GET /health
POST /predict
入力例:
{"atr14": 12.3}

出力例:
{"p_move": 0.73, "decision": 1, "decision_threshold": 0.6, "model_version": "v1"}

今回の学習結果（あなたのCSVで実行した値）
ROC-AUC: 0.747
metrics:
[
  {
    "threshold": 0.5,
    "accuracy": 0.7116182572614108,
    "precision": 0.802130898021309,
    "recall": 0.7807407407407407,
    "confusion_matrix": [
      [
        159,
        130
      ],
      [
        148,
        527
      ]
    ],
    "positive_rate_pred": 0.6815352697095436,
    "positive_rate_true": 0.700207468879668
  },
  {
    "threshold": 0.6,
    "accuracy": 0.6607883817427386,
    "precision": 0.8640167364016736,
    "recall": 0.6118518518518519,
    "confusion_matrix": [
      [
        224,
        65
      ],
      [
        262,
        413
      ]
    ],
    "positive_rate_pred": 0.495850622406639,
    "positive_rate_true": 0.700207468879668
  },
  {
    "threshold": 0.7,
    "accuracy": 0.6120331950207469,
    "precision": 0.8829516539440203,
    "recall": 0.5140740740740741,
    "confusion_matrix": [
      [
        243,
        46
      ],
      [
        328,
        347
      ]
    ],
    "positive_rate_pred": 0.40767634854771784,
    "positive_rate_true": 0.700207468879668
  }
]
