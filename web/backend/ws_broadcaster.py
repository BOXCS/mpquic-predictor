"""
web/backend/ws_broadcaster.py — WebSocket Real-Time Broadcaster

Maintains a registry of connected WebSocket clients and broadcasts
pre-computed metric/prediction snapshots to all of them.

This module contains NO business logic. It only:
  1. Accepts WebSocket connections at /ws
  2. Adds clients to the active-connection registry
  3. Broadcasts JSON payloads assembled from pre-fetched route data
  4. Removes clients on disconnect
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from web.backend.deps import get_metrics_db
from web.backend.routes.metrics import get_latest_metrics, get_switching_events
from web.backend.routes.predictions import get_latest_prediction, get_active_path

logger = logging.getLogger("ws_broadcaster")

router = APIRouter(tags=["websocket"])

# ── Connection registry ────────────────────────────────────────────────────────
_clients: set[WebSocket] = set()


async def _add(ws: WebSocket) -> None:
    _clients.add(ws)
    logger.info("WS client connected. Total: %d", len(_clients))


async def _remove(ws: WebSocket) -> None:
    _clients.discard(ws)
    logger.info("WS client disconnected. Total: %d", len(_clients))


# ── Public broadcast API ───────────────────────────────────────────────────────

async def broadcast(payload: dict[str, Any]) -> None:
    """Push a JSON payload to every connected WebSocket client."""
    if not _clients:
        return
    message = json.dumps(payload)
    dead: list[WebSocket] = []
    
    snapshot = list(_clients)
    for ws in snapshot:
        try:
            await ws.send_text(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        await _remove(ws)


def connected_count() -> int:
    """Return the number of currently connected WebSocket clients."""
    return len(_clients)


# ── Background Push Loop ───────────────────────────────────────────────────────

async def start_push_loop(interval_sec: float = 1.0) -> None:
    """
    Periodically fetch the latest snapshot using route handlers and
    push it to all connected WebSocket clients. Runs for the application lifetime.
    """
    last_event_id = 0
    
    while True:
        await asyncio.sleep(interval_sec)
        if not _clients:
            continue
            
        try:
            db_gen = get_metrics_db()
            db = next(db_gen)
            
            try:
                # Fetch data directly from route handlers (no business logic here)
                latest_metrics = get_latest_metrics(limit=2, path_id=None, db=db).get("data", [])
                latest_prediction = get_latest_prediction(db=db).get("data")
                active_path = get_active_path(db=db).get("data")
                events = get_switching_events(limit=50, db=db).get("data", [])
                
                # Filter for new switching events since last push
                new_events = [e for e in events if e["id"] > last_event_id]
                if new_events:
                    last_event_id = max(e["id"] for e in new_events)
                
                payload = {
                    "type": "update",
                    "metrics": latest_metrics,
                    "prediction": latest_prediction,
                    "active_path": active_path,
                    "switching_events": new_events
                }
                
                await broadcast(payload)
            finally:
                try:
                    next(db_gen)
                except StopIteration:
                    pass
        except Exception as exc:
            import traceback
            logger.error("WS push loop CRASHED: %s\n%s", exc, traceback.format_exc())


# WebSocket endpoint is registered in main.py to avoid APIRouter issues
