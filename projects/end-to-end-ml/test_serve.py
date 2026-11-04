"""Optional API smoke test; install the `mlops` extra first."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

import serve  # noqa: E402
from train import train  # noqa: E402


def test_api_reports_missing_model_then_predicts(tmp_path, monkeypatch) -> None:
    model_path = tmp_path / "pipeline.joblib"
    monkeypatch.setattr(serve, "MODEL_PATH", model_path)
    client = TestClient(serve.app)

    assert client.get("/health").json() == {"status": "model_missing"}
    unavailable = client.post(
        "/predict",
        json={
            "tenure_months": 12,
            "monthly_charge": 75.0,
            "support_calls_90d": 1,
            "contract": "monthly",
        },
    )
    assert unavailable.status_code == 503

    train(model_path)
    assert client.get("/health").json() == {"status": "ready"}
    response = client.post(
        "/predict",
        json={
            "tenure_months": 12,
            "monthly_charge": 75.0,
            "support_calls_90d": 1,
            "contract": "monthly",
        },
    )
    assert response.status_code == 200
    assert 0 <= response.json()["synthetic_churn_probability"] <= 1
