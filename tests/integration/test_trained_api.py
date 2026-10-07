"""API contract tests against a fixture-independent set of trained artifacts."""

import json

import pytest
from fastapi.testclient import TestClient

from backend import main
from backend.config import BACKGROUND_PATH, MODEL_PATHS, SCHEMA_PATH, THRESHOLDS_PATH

client = TestClient(main.app)


@pytest.fixture(scope="module", autouse=True)
def require_trained_artifacts():
    required = [SCHEMA_PATH, BACKGROUND_PATH, THRESHOLDS_PATH, *MODEL_PATHS.values()]
    missing = [str(path) for path in required if not path.is_file()]
    assert not missing, "Run `python -m ml.train` before API integration tests; missing: " + ", ".join(missing)


pytestmark = pytest.mark.integration


def synthetic_payload():
    return json.loads((SCHEMA_PATH.parent.parent / "docs" / "sample_request.json").read_text(encoding="utf-8"))


def test_health_contract():
    assert client.get("/health").json() == {"status": "ok"}


def test_schema_matches_saved_model_input_contract():
    response = client.get("/schema")
    assert response.status_code == 200
    names = {item["feature_name"] for item in response.json()["features"]}
    assert names == {item["feature_name"] for item in json.loads(SCHEMA_PATH.read_text())["features"]}
    assert names.isdisjoint({"LAD", "LCX", "RCA", "Cath", "Exertional CP"})


def test_predict_returns_typed_complete_deterministic_response():
    payload = synthetic_payload()
    first, second = client.post("/predict", json=payload), client.post("/predict", json=payload)
    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    body = first.json()
    assert set(body["predictions"]) == {"cad", "lad", "lcx", "rca"}
    assert set(body["explanations"]) == {"cad", "lad", "lcx", "rca"}
    assert body["measurements"] == payload
    assert body["anatomical_localization"] is False
    for result in body["predictions"].values():
        assert 0 <= result["probability"] <= 1
        assert result["risk"] in {"LOW", "MODERATE", "HIGH"}
    assert first.json() == second.json()


def test_predict_rejects_missing_or_invalid_fields():
    payload = synthetic_payload()
    payload.pop("Age")
    assert client.post("/predict", json=payload).status_code == 422
    payload = synthetic_payload()
    payload["Sex"] = "Female"
    assert client.post("/predict", json=payload).status_code == 422


def test_missing_model_is_a_503_not_a_success(monkeypatch):
    def missing(_target):
        raise FileNotFoundError("test model missing")
    monkeypatch.setattr(main, "_model", missing)
    response = client.post("/predict", json=synthetic_payload())
    assert response.status_code == 503
    assert response.json()["detail"] == "test model missing"


def test_cors_allows_configured_frontend_origin_only():
    allowed = client.options("/predict", headers={
        "Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST",
    })
    assert allowed.headers.get("access-control-allow-origin") == "http://localhost:5173"
    rejected = client.options("/predict", headers={
        "Origin": "https://untrusted.example", "Access-Control-Request-Method": "POST",
    })
    assert "access-control-allow-origin" not in rejected.headers


def test_openapi_documents_prediction_response():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    predict = response.json()["paths"]["/predict"]["post"]["responses"]["200"]
    assert predict["content"]["application/json"]["schema"]["$ref"].endswith("PredictResponse")
