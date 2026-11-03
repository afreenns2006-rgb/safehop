"""Shared CSV preparation and split logic for SafeHop training commands."""

import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from src.features import FEATURE_NAMES, extract_features
except ModuleNotFoundError:  # Support scripts run directly from the src directory.
    from features import FEATURE_NAMES, extract_features

LABELS = {
    "benign": 0, "legitimate": 0, "good": 0, "safe": 0, "0": 0,
    "phishing": 1, "malicious": 1, "bad": 1, "unsafe": 1, "1": 1,
}


def load_training_data(data_path, url_col, label_col):
    """Load, clean, and label the supported rows without extracting features."""
    data = pd.read_csv(data_path)
    data = data.dropna(subset=[url_col, label_col]).drop_duplicates(subset=[url_col]).copy()
    data["_target"] = data[label_col].astype(str).str.strip().str.lower().map(LABELS)
    data = data.dropna(subset=["_target"])
    data["_target"] = data["_target"].astype(int)
    counts = data["_target"].value_counts().reindex([0, 1], fill_value=0)
    return data[url_col].astype(str), data["_target"], counts


def make_feature_matrix(urls):
    """Extract ordered feature rows from the cleaned URL series."""
    return pd.DataFrame.from_records(urls.map(extract_features), columns=FEATURE_NAMES)


def select_training_features(X, y):
    """Print class means and remove strongly class-skewed format features."""
    targets = y.to_numpy()
    means = {
        label: X.loc[targets == label, ["has_https", "has_port"]].mean()
        for label in (0, 1)
    }
    print(
        "Feature class means (has_https, has_port): "
        f"benign=({means[0]['has_https']:.4f}, {means[0]['has_port']:.4f}), "
        f"phishing=({means[1]['has_https']:.4f}, {means[1]['has_port']:.4f})"
    )
    dropped = []
    for name in ("has_https", "has_port"):
        class_rates = sorted((means[0][name], means[1][name]))
        difference = class_rates[1] - class_rates[0]
        strong_ratio = class_rates[0] == 0 or class_rates[1] >= 5 * class_rates[0]
        if difference >= 0.05 and strong_ratio:
            dropped.append(name)
    if dropped:
        print("Dropping strongly class-skewed features: " + ", ".join(dropped))
    else:
        print("No strongly class-skewed features dropped.")
    return [name for name in FEATURE_NAMES if name not in dropped]


def stratified_split(X, y):
    """Return the consistent random 80/20 stratified split."""
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
