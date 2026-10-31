"""SafeHop FastAPI service."""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.predict import predict_url

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "model.pkl"
app = FastAPI(title="SafeHop API")


class PredictRequest(BaseModel):
    url: str = Field(min_length=3, max_length=2048)


@app.post("/api/predict")
def api_predict(body: PredictRequest):
    if not MODEL_PATH.is_file():
        raise HTTPException(status_code=503, detail="SafeHop model is missing. Train a model first.")
    try:
        return predict_url(body.url, str(MODEL_PATH))
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="SafeHop model is missing. Train a model first.") from None


@app.get("/api/health")
def health():
    return {"status": "ok"}


app.mount("/", StaticFiles(directory=str(ROOT / "frontend"), html=True), name="frontend")
