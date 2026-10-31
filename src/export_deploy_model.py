"""Train and save the compact SafeHop deployment random forest."""

import argparse
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

try:
    from src.features import FEATURE_NAMES
    from src.training_data import load_training_data, make_feature_matrix, stratified_split
except ModuleNotFoundError:  # Support `python src/export_deploy_model.py` from the project root.
    from features import FEATURE_NAMES
    from training_data import load_training_data, make_feature_matrix, stratified_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--url-col", default="url")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--out", default="models/model_deploy.pkl")
    args = parser.parse_args()

    urls, y, counts = load_training_data(args.data, args.url_col, args.label_col)
    print(f"Class counts: benign={counts[0]}, phishing={counts[1]}")
    if counts.min() < 2:
        parser.error("Need at least two rows in each supported class for a stratified split")
    X = make_feature_matrix(urls)
    X_train, X_test, y_train, y_test = stratified_split(X, y)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=25,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }
    print("Random Forest (deployment): " + " ".join(f"{name}={value:.4f}" for name, value in metrics.items()))

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURE_NAMES, "name": "Random Forest"}, out, compress=3)
    print(f"Saved {out} ({out.stat().st_size / (1024 * 1024):.2f} MB)")


if __name__ == "__main__":
    main()
