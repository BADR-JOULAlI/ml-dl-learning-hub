"""Serve the locally trained synthetic churn pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_PATH = Path(os.environ.get("MODEL_PATH", "artifacts/churn_pipeline.joblib"))
app = FastAPI(title="ML Hub Churn Demo", version="0.1.0")


class ChurnFeatures(BaseModel):
    """Validated features expected by the trained pipeline."""

    tenure_months: int = Field(ge=1, le=120)
    monthly_charge: float = Field(gt=0, le=1_000)
    support_calls_90d: int = Field(ge=0, le=100)
    contract: Literal["monthly", "annual", "two_year"]


def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    """Report whether a model artifact is available."""
    return {"status": "ready" if MODEL_PATH.exists() else "model_missing"}


@app.post("/predict")
def predict(features: ChurnFeatures) -> dict[str, float]:
    """Return an illustrative probability from the synthetic model."""
    model = load_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model artifact is not available; train first.")
    row_data = features.model_dump() if hasattr(features, "model_dump") else features.dict()
    row = pd.DataFrame([row_data])
    probability = float(model.predict_proba(row)[0, 1])
    return {"synthetic_churn_probability": probability}
