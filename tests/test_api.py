"""HTTP API tests (no running server; uses Starlette TestClient)."""

from starlette.testclient import TestClient

from src.app.api import app


def test_health_ok():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "ok"
    assert "models" in data
    assert "security" in data


def test_health_has_request_id_header():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert "x-request-id" in {k.lower(): v for k, v in r.headers.items()}


def test_predict_batch_missing_columns():
    client = TestClient(app)
    r = client.post("/predict-batch", json={"records": [{"duration": 0}]})
    assert r.status_code == 400
    assert "error" in r.json()


def test_predict_multiclass_missing_columns():
    client = TestClient(app)
    r = client.post("/predict-multiclass", json={"records": [{}]})
    assert r.status_code == 400

