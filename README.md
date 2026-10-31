# SafeHop

SafeHop is a local phishing and malicious URL risk detector. It extracts URL-only features, trains several scikit-learn classifiers, and serves a lightweight browser interface and JSON API. Submitted URLs are analyzed as text and are never fetched.

## Features

- 25 numeric URL features, including host, path, punctuation, entropy, and risk indicators.
- Model training and comparison across logistic regression, decision tree, random forest, and gradient boosting.
- Cached prediction model, risk score, rule-based reasons, and feature values.
- Responsive single-file web UI with scan feedback and recent-check history.

## Project structure

```text
src/                 Feature extraction, training, and prediction
frontend/index.html  Self-contained interface
tests/               Feature and API tests
data/                Place training CSV files here
models/              Trained model and evaluation outputs
main.py              FastAPI application
requirements.txt     Python dependencies
```

## Setup on Windows

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train

The CSV should have a URL column and a label column. Supported benign labels are `benign`, `legitimate`, `good`, `safe`, and `0`; supported risky labels are `phishing`, `malicious`, `bad`, `unsafe`, and `1`.

```powershell
python src/train.py --data data/<file>.csv --url-col url --label-col type
```

Training writes the selected model, a metrics table, and two charts under `models/`.

## Test

```powershell
pytest
```

## Run

```powershell
uvicorn main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The health endpoint is `GET /api/health`; predictions use `POST /api/predict`.

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/predict -H "Content-Type: application/json" -d "{\"url\":\"https://example.com/login\"}"
```

Risk scores below 35 are **low** (likely legitimate), scores from 35 to below 65 are **medium** (suspicious; proceed with caution), and scores of 65 or higher are **high** (potential phishing URL). Scores are model estimates and should not be treated as a guarantee.
