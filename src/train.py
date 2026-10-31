"""Train and compare URL classifiers."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import joblib

try:
    from src.features import FEATURE_NAMES
    from src.training_data import load_training_data, make_feature_matrix, stratified_split
except ModuleNotFoundError:  # Support `python src/train.py` from the project root.
    from features import FEATURE_NAMES
    from training_data import load_training_data, make_feature_matrix, stratified_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--url-col", default="url")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--out", default="models/model.pkl")
    args = parser.parse_args()
    urls, y, counts = load_training_data(args.data, args.url_col, args.label_col)
    print(f"Class counts: benign={counts[0]}, phishing={counts[1]}")
    if counts.min() < 2:
        parser.error("Need at least two rows in each supported class for a stratified split")
    X = make_feature_matrix(urls)
    X_train, X_test, y_train, y_test = stratified_split(X, y)
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }
    results, fitted = [], {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        matrix = confusion_matrix(y_test, pred, labels=[0, 1])
        metrics = {"model": name, "accuracy": accuracy_score(y_test, pred), "precision": precision_score(y_test, pred, zero_division=0), "recall": recall_score(y_test, pred, zero_division=0), "f1": f1_score(y_test, pred, zero_division=0), "roc_auc": roc_auc_score(y_test, probs), "confusion_matrix": str(matrix.tolist())}
        print(f"{name}: accuracy={metrics['accuracy']:.4f} precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} F1={metrics['f1']:.4f} ROC-AUC={metrics['roc_auc']:.4f}\nConfusion matrix:\n{matrix}")
        results.append(metrics)
        fitted[name] = (model, matrix)
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out.parent / "results.csv", index=False)
    best_name = max(results, key=lambda item: item["f1"])["model"]
    best, matrix = fitted[best_name]
    joblib.dump({"model": best, "features": FEATURE_NAMES, "name": best_name}, out)
    fig, ax = plt.subplots()
    ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Benign", "Phishing"], yticklabels=["Benign", "Phishing"], xlabel="Predicted", ylabel="Actual", title=f"Confusion Matrix: {best_name}")
    for (i, j), value in __import__("numpy").ndenumerate(matrix):
        ax.text(j, i, str(value), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out.parent / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    if hasattr(best, "feature_importances_"):
        importance = best.feature_importances_
    elif hasattr(best, "coef_"):
        importance = abs(best.coef_[0])
    else:
        importance = abs(best[-1].coef_[0])
    order = __import__("numpy").argsort(importance)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh([FEATURE_NAMES[i] for i in order], importance[order])
    ax.set(title=f"Feature Importance: {best_name}", xlabel="Importance")
    fig.tight_layout()
    fig.savefig(out.parent / "feature_importance.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
