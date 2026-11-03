import pandas as pd

from src.training_data import select_training_features


def test_select_training_features_drops_strongly_class_skewed_indicator(capsys):
    X = pd.DataFrame({
        "has_https": [0] * 10 + [1] * 2,
        "has_port": [0] * 12,
        "url_length": list(range(12)),
    })
    y = pd.Series([0] * 10 + [1] * 2)

    features = select_training_features(X, y)

    assert "has_https" not in features
    assert "has_port" in features
    assert "url_length" in features
    output = capsys.readouterr().out
    assert "benign=(0.0000, 0.0000)" in output
    assert "phishing=(1.0000, 0.0000)" in output
    assert "Dropping strongly class-skewed features: has_https" in output
