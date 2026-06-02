"""
web/backend/routes/metrics.py — Metrics REST Endpoints

Exposes RTT, goodput, and switching-event history from metrics.db.
All endpoints are read-only; all responses conform to the standard
{ "status", "data", "message" } envelope.

Endpoint prefix: /metrics
"""

from __future__ import annotations

import statistics
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from web.backend.deps import get_metrics_db
from server.db_writer import NetworkMetric, PredictionResult, SwitchingEvent

router = APIRouter(prefix="/metrics", tags=["metrics"])


# ── Pydantic response models ───────────────────────────────────────────────────

class MetricItem(BaseModel):
    id: int
    timestamp: str
    path_id: int
    rtt_ms: float
    goodput_bps: float
    loss_pct: float

    model_config = {"from_attributes": True}


class PathStats(BaseModel):
    path_id: int
    path_label: str
    avg_rtt_ms: float
    min_rtt_ms: float
    max_rtt_ms: float
    avg_goodput_bps: float
    avg_loss_pct: float
    sample_count: int


class MetricsSummaryData(BaseModel):
    window_minutes: int
    per_path: List[PathStats]
    total_samples: int


class SwitchingEventItem(BaseModel):
    id: int
    timestamp: str
    from_path: int
    to_path: int
    reason: str

    model_config = {"from_attributes": True}


# ── Helpers ────────────────────────────────────────────────────────────────────

_PATH_LABELS = {1: "wlan0", 2: "eth0"}


def _metric_to_item(row: NetworkMetric) -> MetricItem:
    return MetricItem(
        id=row.id,
        timestamp=row.timestamp.isoformat(),
        path_id=row.path_id,
        rtt_ms=round(row.rtt_ms, 3),
        goodput_bps=round(row.goodput_bps, 3),
        loss_pct=round(row.loss_pct, 3),
    )


def _event_to_item(row: SwitchingEvent) -> SwitchingEventItem:
    return SwitchingEventItem(
        id=row.id,
        timestamp=row.timestamp.isoformat(),
        from_path=row.from_path,
        to_path=row.to_path,
        reason=row.reason,
    )


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.get("/", summary="Latest RTT and goodput per path")
def get_latest_metrics(
    limit: int = Query(
        default=50,
        ge=1,
        le=500,
        description="Maximum number of rows to return (most recent first)",
    ),
    path_id: Optional[int] = Query(
        default=None,
        ge=1,
        le=2,
        description="Filter by path ID: 1 = wlan0, 2 = eth0",
    ),
    db: Session = Depends(get_metrics_db),
) -> dict:
    """
    Return the most recent network metric records (RTT, goodput, loss).
    Optionally filter by path ID. Results are ordered newest-first.
    """
    query = db.query(NetworkMetric).order_by(desc(NetworkMetric.timestamp))
    if path_id is not None:
        query = query.filter(NetworkMetric.path_id == path_id)
    rows = query.limit(limit).all()
    return {
        "status": "ok",
        "data": [_metric_to_item(r).model_dump() for r in rows],
        "message": f"{len(rows)} metric record(s) returned",
    }


@router.get("/summary", summary="Aggregated RTT and goodput stats over a time window")
def get_metrics_summary(
    window_minutes: int = Query(
        default=10,
        ge=1,
        le=1440,
        description="Rolling time window in minutes (default: last 10 minutes)",
    ),
    db: Session = Depends(get_metrics_db),
) -> dict:
    """
    Return per-path aggregated statistics (avg/min/max RTT, avg goodput,
    avg loss) computed over the most recent `window_minutes` of data.
    """
    since = datetime.utcnow() - timedelta(minutes=window_minutes)
    rows = (
        db.query(NetworkMetric)
        .filter(NetworkMetric.timestamp >= since)
        .order_by(NetworkMetric.path_id, NetworkMetric.timestamp)
        .all()
    )

    # Group by path
    by_path: dict[int, list[NetworkMetric]] = {}
    for r in rows:
        by_path.setdefault(r.path_id, []).append(r)

    per_path_stats: list[dict] = []
    for pid, path_rows in sorted(by_path.items()):
        rtts = [r.rtt_ms for r in path_rows]
        goodputs = [r.goodput_bps for r in path_rows]
        losses = [r.loss_pct for r in path_rows]
        per_path_stats.append(
            PathStats(
                path_id=pid,
                path_label=_PATH_LABELS.get(pid, f"path{pid}"),
                avg_rtt_ms=round(statistics.mean(rtts), 3),
                min_rtt_ms=round(min(rtts), 3),
                max_rtt_ms=round(max(rtts), 3),
                avg_goodput_bps=round(statistics.mean(goodputs), 3),
                avg_loss_pct=round(statistics.mean(losses), 3),
                sample_count=len(path_rows),
            ).model_dump()
        )

    summary = MetricsSummaryData(
        window_minutes=window_minutes,
        per_path=per_path_stats,
        total_samples=len(rows),
    )
    return {
        "status": "ok",
        "data": summary.model_dump(),
        "message": (
            f"Aggregated stats for the last {window_minutes} minute(s): "
            f"{len(rows)} sample(s) across {len(by_path)} path(s)"
        ),
    }


@router.get("/switching-events", summary="History of path switching events")
def get_switching_events(
    limit: int = Query(default=20, ge=1, le=200),
    db: Session = Depends(get_metrics_db),
) -> dict:
    """Return the most recent path switching events, newest first."""
    rows = (
        db.query(SwitchingEvent)
        .order_by(desc(SwitchingEvent.timestamp))
        .limit(limit)
        .all()
    )
    return {
        "status": "ok",
        "data": [_event_to_item(r).model_dump() for r in rows],
        "message": f"{len(rows)} switching event(s) returned",
    }
