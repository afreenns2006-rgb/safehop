from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_empty_body():
    assert client.post("/api/predict", json={}).status_code == 422


def test_predict_short_url():
    assert client.post("/api/predict", json={"url": "ab"}).status_code == 422
