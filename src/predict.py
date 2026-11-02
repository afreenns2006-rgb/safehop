"""Load a trained model and produce URL risk assessments."""

from functools import lru_cache
import logging
from pathlib import Path
import sys

import joblib
import pandas as pd

from src.features import FEATURE_NAMES, extract_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelFileMissingError(FileNotFoundError):
    """Raised only when the selected model artifact does not exist."""


class ModelOperationError(RuntimeError):
    """Wrap a logged model loading or prediction failure for the API."""

    def __init__(self, exception_type):
        self.exception_type = exception_type
        super().__init__(exception_type)


def get_model_path():
    """Choose the deployment artifact when available, otherwise the standard model."""
    deployment_model = PROJECT_ROOT / "models" / "model_deploy.pkl"
    return deployment_model if deployment_model.exists() else PROJECT_ROOT / "models" / "model.pkl"


@lru_cache(maxsize=8)
def _load_model(model_path):
    return joblib.load(model_path)


def predict_url(url, model_path=None):
    """Predict phishing probability and explain rule-based indicators."""
    path = Path(model_path).resolve() if model_path is not None else get_model_path().resolve()
    try:
        if not path.exists():
            raise ModelFileMissingError(str(path))
        artifact = _load_model(str(path))
        features = extract_features(url)
        model = artifact["model"]
        row = pd.DataFrame([[features[name] for name in FEATURE_NAMES]], columns=FEATURE_NAMES)
        score = round(float(model.predict_proba(row)[0][1]) * 100, 1)
        level, label = ("low", "Low Risk - Likely legitimate") if score < 35 else (("medium", "Medium Risk - Suspicious, proceed with caution") if score < 65 else ("high", "High Risk - Potential phishing URL"))
        reasons = []
        checks = [(features["has_ip"], "IP address used as host"), (features["num_at"] > 0, "@ symbol present"), (not features["has_https"], "No HTTPS"), (features["num_subdomains"] >= 3, "3 or more subdomains"), (features["url_length"] > 75, "URL longer than 75 characters"), (features["keyword_count"] >= 2, "2 or more suspicious keywords"), (features["is_shortener"], "URL shortener detected"), (features["tld_suspicious"], "Risky top-level domain"), (features["has_punycode"], "Punycode hostname"), (features["num_hyphens"] >= 4, "4 or more hyphens")]
        reasons.extend(reason for matched, reason in checks if matched)
        return {"url": url, "score": score, "level": level, "label": label, "reasons": reasons, "features": features}
    except ModelFileMissingError:
        raise
    except Exception as exc:
        logger.exception("Model loading or prediction failed for %s", path)
        raise ModelOperationError(type(exc).__name__) from exc
