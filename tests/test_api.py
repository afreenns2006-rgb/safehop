import logging

from fastapi.testclient import TestClient

import main
from main import app
from src import predict as predictor

client = TestClient(app)


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert isinstance(payload["model_loaded"], bool)
    assert payload["model_file"] == main.MODEL_PATH.name


def test_predict_empty_body():
    assert client.post("/api/predict", json={}).status_code == 422


def test_predict_short_url():
    assert client.post("/api/predict", json={"url": "ab"}).status_code == 422


def test_predict_missing_model_returns_503(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "MODEL_PATH", tmp_path / "missing.pkl")
    response = client.post("/api/predict", json={"url": "https://example.com"})
    assert response.status_code == 503
    assert response.json()["detail"] == "Model file not found: missing.pkl"


def test_predict_other_error_returns_500_and_logs_traceback(monkeypatch, caplog):
    def broken_prediction(url, model_path):
        raise ValueError("broken model")

    monkeypatch.setattr(main, "predict_url", broken_prediction)
    with caplog.at_level(logging.ERROR):
        response = client.post("/api/predict", json={"url": "https://example.com"})
    assert response.status_code == 500
    assert response.json()["detail"] == "Model could not be loaded: ValueError"
    assert "Unexpected model prediction failure" in caplog.text
    assert "ValueError: broken model" in caplog.text


def test_predict_model_load_error_logs_traceback_and_preserves_exception_type(monkeypatch, tmp_path, caplog):
    model_file = tmp_path / "corrupt.pkl"
    model_file.write_bytes(b"corrupt")

    def broken_load(path):
        raise ValueError("invalid pickle")

    monkeypatch.setattr(predictor, "_load_model", broken_load)
    with caplog.at_level(logging.ERROR):
        try:
            predictor.predict_url("https://example.com", str(model_file))
        except predictor.ModelOperationError as exc:
            assert exc.exception_type == "ValueError"
        else:
            raise AssertionError("Expected ModelOperationError")
    assert "Model loading or prediction failed" in caplog.text
    assert "ValueError: invalid pickle" in caplog.text
