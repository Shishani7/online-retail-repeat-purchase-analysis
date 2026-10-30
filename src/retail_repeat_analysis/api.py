"""Validated HTTP scoring interface for the committed, trusted model artifact."""
import hashlib
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from .modeling import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[2]
Count = Annotated[int, Field(strict=True, ge=0)]
PositiveCount = Annotated[int, Field(strict=True, ge=1)]
Nonnegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]


class CustomerFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    recency_days: Count
    frequency_orders: PositiveCount
    monetary_total: Positive
    avg_order_value: Positive
    max_order_value: Positive
    avg_items_per_order: Positive
    active_months: PositiveCount
    unique_products: PositiveCount
    total_units: Positive
    avg_days_between_orders: Nonnegative | None = None
    customer_tenure_days: Count
    country: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]

    @model_validator(mode="after")
    def coherent_history(self):
        if self.recency_days > self.customer_tenure_days:
            raise ValueError("Recency cannot exceed customer tenure")
        if self.active_months > self.frequency_orders:
            raise ValueError("Active months cannot exceed order count")
        if self.avg_order_value > self.max_order_value or self.max_order_value > self.monetary_total:
            raise ValueError("Require average order value <= maximum order value <= total spend")
        return self


class Prediction(BaseModel):
    score: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    prediction: int
    threshold: float
    horizon_days: int
    model_name: str
    model_version: str


def load_bundle(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    model_path = directory / "repeat_purchase.joblib"
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != metadata["sha256"]:
        raise ValueError("Model checksum does not match metadata")
    if metadata["feature_columns"] != FEATURE_COLUMNS:
        raise ValueError("Model feature contract does not match the application")
    # This artifact is shipped by the repository owner; never load arbitrary uploaded joblib files.
    return joblib.load(model_path), metadata


def create_app(model_directory=None):
    directory = Path(model_directory or os.environ.get("MODEL_DIR", ROOT / "models"))

    @asynccontextmanager
    async def lifespan(app):
        app.state.model, app.state.metadata = load_bundle(directory)
        yield

    app = FastAPI(title="Online Retail Repeat Purchase Scoring", version="0.2.0", lifespan=lifespan,
                  description="Educational existing-customer scoring. Scores are not calibrated probabilities or causal uplift.")

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request, error):
        # Do not echo raw inputs: non-standard JSON NaN/Infinity cannot be serialized in a JSON response.
        detail = [{key: item[key] for key in ["loc", "msg", "type"]} for item in error.errors()]
        return JSONResponse(status_code=422, content={"detail": detail})

    @app.get("/health")
    def health():
        return {"status": "ok", "model_version": app.state.metadata["model_version"]}

    @app.get("/metadata")
    def metadata():
        info = app.state.metadata
        return {key: info[key] for key in ["model_name", "model_version", "horizon_days", "threshold", "feature_columns", "feature_contract"]}

    @app.post("/predict", response_model=Prediction)
    def predict(features: CustomerFeatures):
        frame = pd.DataFrame([features.model_dump()], columns=FEATURE_COLUMNS)
        score = float(app.state.model.predict_proba(frame)[0, 1])
        info = app.state.metadata
        return Prediction(score=score, prediction=int(score >= info["threshold"]),
                          threshold=info["threshold"], horizon_days=info["horizon_days"],
                          model_name=info["model_name"], model_version=info["model_version"])

    return app


app = create_app()
