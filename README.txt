Deploy:
- gcloud run deploy --source . --region asia-northeast1 --allow-unauthenticated

Endpoints:
POST /predict  { "atr14": 3.2 }
POST /direction {}
