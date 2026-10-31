"""Load a trained model and produce URL risk assessments."""

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from src.features import FEATURE_NAMES, extract_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=8)
def _load_model(model_path):
    return joblib.load(model_path)


def predict_url(url, model_path=None):
    """Predict phishing probability and explain rule-based indicators."""
    if model_path is None:
        deployment_model = PROJECT_ROOT / "models" / "model_deploy.pkl"
        model_path = deployment_model if deployment_model.is_file() else PROJECT_ROOT / "models" / "model.pkl"
    artifact = _load_model(str(Path(model_path).resolve()))
    features = extract_features(url)
    model = artifact["model"]
    row = pd.DataFrame([[features[name] for name in FEATURE_NAMES]], columns=FEATURE_NAMES)
    score = round(float(model.predict_proba(row)[0][1]) * 100, 1)
    level, label = ("low", "Low Risk - Likely legitimate") if score < 35 else (("medium", "Medium Risk - Suspicious, proceed with caution") if score < 65 else ("high", "High Risk - Potential phishing URL"))
    reasons = []
    checks = [(features["has_ip"], "IP address used as host"), (features["num_at"] > 0, "@ symbol present"), (not features["has_https"], "No HTTPS"), (features["num_subdomains"] >= 3, "3 or more subdomains"), (features["url_length"] > 75, "URL longer than 75 characters"), (features["keyword_count"] >= 2, "2 or more suspicious keywords"), (features["is_shortener"], "URL shortener detected"), (features["tld_suspicious"], "Risky top-level domain"), (features["has_punycode"], "Punycode hostname"), (features["num_hyphens"] >= 4, "4 or more hyphens")]
    reasons.extend(reason for matched, reason in checks if matched)
    return {"url": url, "score": score, "level": level, "label": label, "reasons": reasons, "features": features}
