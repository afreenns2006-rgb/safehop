"""SafeHop FastAPI service."""

import logging
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import sys

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.predict import (
    ModelFileMissingError,
    ModelOperationError,
    _load_model,
    get_model_path,
    predict_url,
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = get_model_path()
MODEL_LOADED = False
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(title="SafeHop API")


class PredictRequest(BaseModel):
    url: str = Field(min_length=3, max_length=2048)


@app.on_event("startup")
def startup():
    """Log runtime details and try loading the model without blocking startup."""
    global MODEL_LOADED
    path = MODEL_PATH.resolve()
    exists = path.exists()
    size_mb = path.stat().st_size / (1024 * 1024) if path.is_file() else 0.0
    logger.info("Model path: %s | exists: %s | size: %.2f MB", path, exists, size_mb)
    packages = ("scikit-learn", "numpy", "pandas", "joblib")
    installed = []
    for package in packages:
        try:
            package_version = version(package)
        except PackageNotFoundError:
            package_version = "not installed"
        installed.append(f"{package}={package_version}")
    logger.info("Runtime versions: python=%s %s", platform.python_version(), " ".join(installed))
    try:
        _load_model(str(path))
        MODEL_LOADED = True
        logger.info("Model loaded OK")
    except Exception:
        MODEL_LOADED = False
        logger.exception("Model load failed during startup")


@app.post("/api/predict")
def api_predict(body: PredictRequest):
    global MODEL_LOADED
    try:
        result = predict_url(body.url, str(MODEL_PATH))
        MODEL_LOADED = True
        return result
    except ModelFileMissingError:
        raise HTTPException(status_code=503, detail=f"Model file not found: {MODEL_PATH.name}") from None
    except ModelOperationError as exc:
        raise HTTPException(status_code=500, detail=f"Model could not be loaded: {exc.exception_type}") from None
    except Exception as exc:
        logger.exception("Unexpected model prediction failure")
        raise HTTPException(status_code=500, detail=f"Model could not be loaded: {type(exc).__name__}") from None


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_LOADED, "model_file": MODEL_PATH.name}


app.mount("/", StaticFiles(directory=str(ROOT / "frontend"), html=True), name="frontend")
