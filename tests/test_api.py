"""Artifact-independent API liveness and contract tests."""

from fastapi.testclient import TestClient

from backend import main

client = TestClient(main.app)


def test_health_contract_is_independent_of_model_artifacts():
    assert client.get("/health").json() == {"status": "ok"}


def test_openapi_documents_typed_prediction_response():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    operation = response.json()["paths"]["/predict"]["post"]
    assert operation["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith("PredictResponse")


def test_schema_reports_unavailable_artifact_as_503(monkeypatch):
    def unavailable():
        raise FileNotFoundError("test schema missing")
    monkeypatch.setattr(main, "get_feature_schema", unavailable)
    response = client.get("/schema")
    assert response.status_code == 503
    assert response.json()["detail"] == "test schema missing"


def test_predict_reports_missing_schema_artifact_as_503(monkeypatch):
    import json
    from pathlib import Path

    def unavailable():
        raise FileNotFoundError("test schema missing")
    monkeypatch.setattr(main, "get_feature_schema", unavailable)
    payload = json.loads((Path(__file__).parents[1] / "docs" / "sample_request.json").read_text())
    response = client.post("/predict", json=payload)
    assert response.status_code == 503
    assert response.json()["detail"] == "test schema missing"


def test_cors_restricts_to_configured_frontend_origins():
    allowed = client.options("/predict", headers={
        "Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST",
    })
    assert allowed.headers.get("access-control-allow-origin") == "http://localhost:5173"
    rejected = client.options("/predict", headers={
        "Origin": "https://untrusted.example", "Access-Control-Request-Method": "POST",
    })
    assert "access-control-allow-origin" not in rejected.headers
