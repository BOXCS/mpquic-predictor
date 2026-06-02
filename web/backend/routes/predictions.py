"""
web/backend/routes/predictions.py — LSTM Prediction REST Endpoints

Exposes LSTM degradation prediction results stored in metrics.db.
All endpoints are read-only. Responses conform to the standard
{ "status", "data", "message" } envelope with Pydantic-validated shapes.

Endpoint prefix: /predictions
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy import desc
from sqlalchemy.orm import Session

from web.backend.deps import get_metrics_db
from server.db_writer import NetworkMetric, PredictionResult, SwitchingEvent

router = APIRouter(prefix="/predictions", tags=["predictions"])


# ── Pydantic response models ───────────────────────────────────────────────────

class PredictionItem(BaseModel):
    id: int
    timestamp: str
    predicted_path: int
    confidence: float
    degradation_detected: bool

    model_config = {"from_attributes": True}


class PathHealthState(BaseModel):
    """Health state derived from the most recent metric and prediction for a path."""
    path_id: int
    path_label: str
    latest_rtt_ms: Optional[float]
    latest_goodput_bps: Optional[float]
    latest_loss_pct: Optional[float]
    degradation_detected: Optional[bool]
    prediction_confidence: Optional[float]
    data_timestamp: Optional[str]


class ActivePathData(BaseModel):
    active_path: int
    path_label: str
    health: List[PathHealthState]


# ── Helpers ────────────────────────────────────────────────────────────────────

_PATH_LABELS = {1: "wlan0", 2: "eth0"}


def _to_item(row: PredictionResult) -> PredictionItem:
    return PredictionItem(
        id=row.id,
        timestamp=row.timestamp.isoformat(),
        predicted_path=row.predicted_path,
        confidence=round(row.confidence, 4),
        degradation_detected=row.degradation_detected,
    )


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.get("/latest", summary="Most recent LSTM prediction result")
def get_latest_prediction(
    db: Session = Depends(get_metrics_db),
) -> dict:
    """Return the single most recent LSTM degradation prediction."""
    row = (
        db.query(PredictionResult)
        .order_by(desc(PredictionResult.timestamp))
        .first()
    )
    if row is None:
        return {
            "status": "ok",
            "data": None,
            "message": "No predictions recorded yet",
        }
    return {
        "status": "ok",
        "data": _to_item(row).model_dump(),
        "message": "Latest LSTM prediction result",
    }


@router.get("/history", summary="Paginated LSTM prediction history")
def get_prediction_history(
    limit: int = Query(
        default=50,
        ge=1,
        le=500,
        description="Maximum number of prediction records to return",
    ),
    offset: int = Query(
        default=0,
        ge=0,
        description="Number of records to skip (pagination offset)",
    ),
    degraded_only: bool = Query(
        default=False,
        description="When true, return only records where degradation was detected",
    ),
    db: Session = Depends(get_metrics_db),
) -> dict:
    """
    Return paginated LSTM prediction history, newest first.
    Supports `degraded_only` to filter for degradation-positive predictions.
    """
    query = db.query(PredictionResult).order_by(desc(PredictionResult.timestamp))
    if degraded_only:
        query = query.filter(PredictionResult.degradation_detected == True)  # noqa: E712
    total = query.count()
    rows = query.offset(offset).limit(limit).all()
    return {
        "status": "ok",
        "data": {
            "items": [_to_item(r).model_dump() for r in rows],
            "total": total,
            "limit": limit,
            "offset": offset,
        },
        "message": f"{len(rows)} of {total} prediction record(s) returned",
    }


@router.get("/active-path", summary="Current active path and per-path health state")
def get_active_path(
    db: Session = Depends(get_metrics_db),
) -> dict:
    """
    Return the currently active path (inferred from the last switching event)
    along with the most recent RTT, goodput, and prediction confidence per path.
    Falls back to path 1 (wlan0) if no switching events have been recorded.
    """
    # Determine active path from most recent switching event
    latest_event = (
        db.query(SwitchingEvent)
        .order_by(desc(SwitchingEvent.timestamp))
        .first()
    )
    active_path = latest_event.to_path if latest_event else 1

    # Collect per-path health: latest metric row + latest prediction for each path
    path_health: list[dict] = []
    for pid in (1, 2):
        latest_metric = (
            db.query(NetworkMetric)
            .filter(NetworkMetric.path_id == pid)
            .order_by(desc(NetworkMetric.timestamp))
            .first()
        )
        latest_pred = (
            db.query(PredictionResult)
            .filter(PredictionResult.predicted_path == pid)
            .order_by(desc(PredictionResult.timestamp))
            .first()
        )
        state = PathHealthState(
            path_id=pid,
            path_label=_PATH_LABELS.get(pid, f"path{pid}"),
            latest_rtt_ms=round(latest_metric.rtt_ms, 3) if latest_metric else None,
            latest_goodput_bps=round(latest_metric.goodput_bps, 3) if latest_metric else None,
            latest_loss_pct=round(latest_metric.loss_pct, 3) if latest_metric else None,
            degradation_detected=latest_pred.degradation_detected if latest_pred else None,
            prediction_confidence=round(latest_pred.confidence, 4) if latest_pred else None,
            data_timestamp=latest_metric.timestamp.isoformat() if latest_metric else None,
        )
        path_health.append(state.model_dump())

    active_label = _PATH_LABELS.get(active_path, f"path{active_path}")
    result = ActivePathData(
        active_path=active_path,
        path_label=active_label,
        health=path_health,
    )

    switch_note = (
        f"Last switch at {latest_event.timestamp.isoformat()}"
        if latest_event
        else "No switching events recorded; defaulting to path 1 (wlan0)"
    )
    return {
        "status": "ok",
        "data": result.model_dump(),
        "message": switch_note,
    }
