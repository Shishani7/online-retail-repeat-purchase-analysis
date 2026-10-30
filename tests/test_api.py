import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from retail_repeat_analysis.api import create_app, load_bundle
from retail_repeat_analysis.modeling import NUMERIC_FEATURES

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def payload():
    return json.loads((ROOT / "examples/predict_request.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(ROOT / "models")) as client:
        yield client


def test_real_model_and_saved_example(client, payload):
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    result = response.json()
    saved = json.loads((ROOT / "examples/predict_response.json").read_text(encoding="utf-8"))
    assert result["score"] == pytest.approx(saved["score"], abs=1e-12)
    assert result["prediction"] == int(result["score"] >= result["threshold"])
    assert result["horizon_days"] == 90
    assert result["model_version"] == saved["model_version"]
    assert client.get("/health").json()["model_version"] == result["model_version"]


@pytest.mark.parametrize("field", NUMERIC_FEATURES)
def test_rejects_negative_features(client, payload, field):
    payload[field] = -1
    assert client.post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize("field", ["country", "frequency_orders", "monetary_total"])
def test_missing_required_fields(client, payload, field):
    del payload[field]
    assert client.post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize("value", ["abc", 1.5, True, None])
def test_rejects_wrong_count_type(client, payload, value):
    payload["frequency_orders"] = value
    assert client.post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_rejects_nonfinite_numbers(client, payload, value):
    payload["monetary_total"] = value
    response = client.post("/predict", content=json.dumps(payload), headers={"Content-Type": "application/json"})
    assert response.status_code == 422


def test_unknown_country_is_supported(client, payload):
    payload["country"] = "Previously unseen country"
    result = client.post("/predict", json=payload)
    assert result.status_code == 200
    assert 0 <= result.json()["score"] <= 1


def test_nullable_gap_for_one_order(client, payload):
    payload["avg_days_between_orders"] = None
    assert client.post("/predict", json=payload).status_code == 200


def test_rejects_extra_fields(client, payload):
    payload["target_repeat_purchase"] = 1
    assert client.post("/predict", json=payload).status_code == 422


def test_rejects_inconsistent_history(client, payload):
    payload["recency_days"] = payload["customer_tenure_days"] + 1
    assert client.post("/predict", json=payload).status_code == 422


def test_docs_describe_prediction_schema(client):
    schema = client.get("/openapi.json").json()
    assert "/predict" in schema["paths"]
    assert "CustomerFeatures" in schema["components"]["schemas"]


def test_detects_corrupted_model(tmp_path):
    (tmp_path / "metadata.json").write_bytes((ROOT / "models/metadata.json").read_bytes())
    (tmp_path / "repeat_purchase.joblib").write_bytes(b"not a model")
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(tmp_path)

