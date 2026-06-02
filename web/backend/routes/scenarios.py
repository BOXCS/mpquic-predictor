"""
web/backend/routes/scenarios.py — Degradation Scenario REST Endpoints

Exposes the 12 simulation scenario definitions (mirrored from
simulator/data_generator.py) and their simulation status derived from
data/logs/path_log2.csv.  The "live" endpoint returns the most recently
active scenario inferred from the live metric stream in metrics.db.

Endpoint prefix: /scenarios
"""

from __future__ import annotations

import csv
import os
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy import desc
from sqlalchemy.orm import Session

from web.backend.deps import get_metrics_db
from server.db_writer import NetworkMetric

router = APIRouter(prefix="/scenarios", tags=["scenarios"])


# ── Pydantic response models ───────────────────────────────────────────────────

class PathCondition(BaseModel):
    base_delay_ms: int
    jitter_ms: int
    loss_pct: float


class ScenarioItem(BaseModel):
    name: str
    duration_seconds: int
    path1: PathCondition
    path2: PathCondition
    simulated: bool
    description: str


class LiveScenarioData(BaseModel):
    """Estimated current scenario derived from live RTT readings in metrics.db."""
    estimated_scenario: Optional[str]
    path1_rtt_ms: Optional[float]
    path2_rtt_ms: Optional[float]
    note: str


# ── Static scenario catalogue ──────────────────────────────────────────────────
# Mirrors simulator/data_generator.py SCENARIOS list exactly.
# Kept inline to avoid importing simulator code into the web layer
# (code-standards.md: evaluation/ must not import from server/).
# The actual data_generator has 12 entries (not 14 as stated in requirements.txt
# comment; count verified against simulator/data_generator.py).

_CATALOGUE: list[dict] = [
    {
        "name": "normal",
        "duration_seconds": 120,
        "path1": {"base_delay_ms": 20,  "jitter_ms": 5,  "loss_pct": 1.0},
        "path2": {"base_delay_ms": 80,  "jitter_ms": 20, "loss_pct": 5.0},
        "description": "Both paths healthy; baseline operating conditions",
    },
    {
        "name": "path1_degrading",
        "duration_seconds": 120,
        "path1": {"base_delay_ms": 60,  "jitter_ms": 20, "loss_pct": 8.0},
        "path2": {"base_delay_ms": 80,  "jitter_ms": 20, "loss_pct": 5.0},
        "description": "wlan0 RTT rising toward degradation threshold",
    },
    {
        "name": "path1_degraded",
        "duration_seconds": 120,
        "path1": {"base_delay_ms": 150, "jitter_ms": 40, "loss_pct": 20.0},
        "path2": {"base_delay_ms": 80,  "jitter_ms": 20, "loss_pct": 5.0},
        "description": "wlan0 severely degraded; eth0 remains healthy",
    },
    {
        "name": "path2_degrading",
        "duration_seconds": 120,
        "path1": {"base_delay_ms": 20,  "jitter_ms": 5,  "loss_pct": 1.0},
        "path2": {"base_delay_ms": 120, "jitter_ms": 35, "loss_pct": 12.0},
        "description": "eth0 RTT rising toward degradation threshold",
    },
    {
        "name": "path2_degraded",
        "duration_seconds": 120,
        "path1": {"base_delay_ms": 20,  "jitter_ms": 5,  "loss_pct": 1.0},
        "path2": {"base_delay_ms": 200, "jitter_ms": 50, "loss_pct": 25.0},
        "description": "eth0 severely degraded; wlan0 remains healthy",
    },
    {
        "name": "both_congested",
        "duration_seconds": 120,
        "path1": {"base_delay_ms": 100, "jitter_ms": 30, "loss_pct": 15.0},
        "path2": {"base_delay_ms": 180, "jitter_ms": 50, "loss_pct": 20.0},
        "description": "Both paths congested; wlan0 less degraded",
    },
    {
        "name": "both_congested_severe",
        "duration_seconds": 120,
        "path1": {"base_delay_ms": 200, "jitter_ms": 60, "loss_pct": 30.0},
        "path2": {"base_delay_ms": 250, "jitter_ms": 70, "loss_pct": 35.0},
        "description": "Both paths severely congested",
    },
    {
        "name": "recovery_path1",
        "duration_seconds": 120,
        "path1": {"base_delay_ms": 25,  "jitter_ms": 8,  "loss_pct": 2.0},
        "path2": {"base_delay_ms": 90,  "jitter_ms": 25, "loss_pct": 6.0},
        "description": "wlan0 recovering from prior degradation",
    },
    {
        "name": "recovery_path2",
        "duration_seconds": 120,
        "path1": {"base_delay_ms": 20,  "jitter_ms": 5,  "loss_pct": 1.0},
        "path2": {"base_delay_ms": 85,  "jitter_ms": 22, "loss_pct": 5.0},
        "description": "eth0 recovering from prior degradation",
    },
    {
        "name": "normal_stable",
        "duration_seconds": 120,
        "path1": {"base_delay_ms": 18,  "jitter_ms": 3,  "loss_pct": 0.5},
        "path2": {"base_delay_ms": 75,  "jitter_ms": 15, "loss_pct": 3.0},
        "description": "Stable baseline with minimal jitter",
    },
    {
        "name": "gradual_degradation_path1",
        "duration_seconds": 180,
        "path1": {"base_delay_ms": 20,  "jitter_ms": 5,  "loss_pct": 1.0},
        "path2": {"base_delay_ms": 80,  "jitter_ms": 20, "loss_pct": 5.0},
        "description": "wlan0 gradually degrades from 60 s to 120 s (target 180 ms delay)",
    },
    {
        "name": "gradual_degradation_path2",
        "duration_seconds": 180,
        "path1": {"base_delay_ms": 20,  "jitter_ms": 5,  "loss_pct": 1.0},
        "path2": {"base_delay_ms": 80,  "jitter_ms": 20, "loss_pct": 5.0},
        "description": "eth0 gradually degrades from 60 s to 120 s (target 220 ms delay)",
    },
]

_SCENARIO_BY_NAME: dict[str, dict] = {s["name"]: s for s in _CATALOGUE}

# ── CSV log path (path_log2.csv written by logger/path_logger.py) ───────────
# The legacy path_log.csv has no scenario column; we check path_log2.csv first.
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
)
_LOG_PATHS = [
    os.path.join(_PROJECT_ROOT, "data", "logs", "path_log2.csv"),
    os.path.join(_PROJECT_ROOT, "data", "logs", "path_log.csv"),
]


def _simulated_scenario_names() -> set[str]:
    """
    Return scenario names found in any CSV log file.
    The logger writes a `scenario` column when available; if absent the CSV
    only confirms that *some* data was collected (no per-scenario status).
    """
    names: set[str] = set()
    for log_path in _LOG_PATHS:
        if not os.path.exists(log_path):
            continue
        try:
            with open(log_path, newline="") as f:
                first_line = f.readline().strip()
                f.seek(0)
                # Detect if there is a header row
                if "scenario" in first_line.lower():
                    reader = csv.DictReader(f)
                    for row in reader:
                        sc = row.get("scenario", "").strip()
                        if sc:
                            names.add(sc)
                else:
                    # Headerless CSV: file exists but no per-scenario info.
                    # Mark all scenarios as simulated when any log data exists.
                    if sum(1 for _ in f) > 0:
                        names.update(s["name"] for s in _CATALOGUE)
        except Exception:
            pass
    return names


def _build_item(entry: dict, simulated: bool) -> ScenarioItem:
    return ScenarioItem(
        name=entry["name"],
        duration_seconds=entry["duration_seconds"],
        path1=PathCondition(**entry["path1"]),
        path2=PathCondition(**entry["path2"]),
        simulated=simulated,
        description=entry["description"],
    )


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.get("/", summary="All degradation scenarios with simulation status")
def list_scenarios() -> dict:
    """
    Return all simulation scenarios with their definition and a
    `simulated` flag indicating whether log data exists for that scenario.
    """
    simulated = _simulated_scenario_names()
    items = [_build_item(s, s["name"] in simulated) for s in _CATALOGUE]
    n_simulated = sum(1 for i in items if i.simulated)
    return {
        "status": "ok",
        "data": [i.model_dump() for i in items],
        "message": f"{n_simulated}/{len(items)} scenarios have simulation data",
    }


@router.get("/live", summary="Estimated current live scenario from the metric stream")
def get_live_scenario(
    db: Session = Depends(get_metrics_db),
) -> dict:
    """
    Estimate which named scenario most closely matches the current live RTT
    readings for each path.  Matches by finding the catalogue entry whose
    `base_delay_ms` values are closest to the observed mean RTT per path.
    Returns None if no live data has been collected yet.
    """
    results: dict[int, float] = {}
    for pid in (1, 2):
        row = (
            db.query(NetworkMetric)
            .filter(NetworkMetric.path_id == pid)
            .order_by(desc(NetworkMetric.timestamp))
            .first()
        )
        if row:
            results[pid] = round(row.rtt_ms, 2)

    if not results:
        data = LiveScenarioData(
            estimated_scenario=None,
            path1_rtt_ms=None,
            path2_rtt_ms=None,
            note="No live metric data available yet",
        )
        return {"status": "ok", "data": data.model_dump(), "message": "No data"}

    p1_rtt = results.get(1)
    p2_rtt = results.get(2)

    # Find best-matching catalogue entry by minimising sum of absolute
    # differences between observed RTT and base_delay_ms per path.
    best_name: Optional[str] = None
    best_score = float("inf")
    for entry in _CATALOGUE:
        score = 0.0
        if p1_rtt is not None:
            score += abs(p1_rtt - entry["path1"]["base_delay_ms"])
        if p2_rtt is not None:
            score += abs(p2_rtt - entry["path2"]["base_delay_ms"])
        if score < best_score:
            best_score = score
            best_name = entry["name"]

    data = LiveScenarioData(
        estimated_scenario=best_name,
        path1_rtt_ms=p1_rtt,
        path2_rtt_ms=p2_rtt,
        note=(
            f"Nearest catalogue match (score={best_score:.1f} ms total deviation). "
            "This is an estimate; the emulator reports the authoritative scenario name."
        ),
    )
    return {
        "status": "ok",
        "data": data.model_dump(),
        "message": f"Estimated live scenario: {best_name}",
    }


@router.get("/{name}", summary="Single scenario definition and status")
def get_scenario(name: str) -> dict:
    """Return the definition and simulation status of a single named scenario."""
    entry = _SCENARIO_BY_NAME.get(name)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"Scenario '{name}' not found. Valid names: {list(_SCENARIO_BY_NAME)}",
        )
    simulated = name in _simulated_scenario_names()
    return {
        "status": "ok",
        "data": _build_item(entry, simulated).model_dump(),
        "message": f"Scenario '{name}' — {'data available' if simulated else 'not yet simulated'}",
    }
