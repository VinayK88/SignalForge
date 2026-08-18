from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .service import build_report, score_event

app = FastAPI(title="SignalForge", version="0.1.0")


class ScoreRequest(BaseModel):
    event_id: str = "runtime-event"
    account_age_days: float = Field(ge=0)
    order_amount: float = Field(ge=0)
    device_accounts_7d: int = Field(ge=0)
    payment_accounts_30d: int = Field(ge=0)
    shipping_accounts_30d: int = Field(ge=0)
    velocity_1h: int = Field(ge=0)
    refund_rate_30d: float = Field(ge=0, le=1)
    distance_from_home_km: float = Field(ge=0)
    new_device: bool = False
    digital_goods: bool = False
    graph_component_size: int = Field(default=4, ge=1)
    risky_neighbor_ratio: float = Field(default=0.05, ge=0, le=1)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "mode": "synthetic"}


@app.post("/score")
def score(request: ScoreRequest) -> dict:
    return score_event(request.model_dump())


@app.get("/report")
def report() -> dict:
    return build_report()
