"""
web/backend/main.py — FastAPI Application Entry Point

Initialises the FastAPI app, registers all route routers and the WebSocket
endpoint, and performs a startup health-check against both SQLite databases.

This module contains NO business logic. Responsibilities:
  - App creation and CORS configuration
  - Route prefix registration (routing only)
  - Lifespan startup check: metrics.db + sensor_data.db connectivity
  - WebSocket registration via ws_broadcaster.router

Run with:
  python -m uvicorn web.backend.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
import os
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

import fastapi
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from web.backend.routes import metrics as metrics_router
from web.backend.routes import predictions as predictions_router
from web.backend.routes import scenarios as scenarios_router
from web.backend import ws_broadcaster

logger = logging.getLogger("web.backend.main")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ── Startup: DB health check ───────────────────────────────────────────────────

def _check_db_connections() -> None:
    """
    Verify both SQLite databases are reachable by executing a lightweight
    query against each. Raises RuntimeError if either check fails.
    """
    from server.db_writer import MetricsSessionLocal, SensorSessionLocal
    from sqlalchemy import text

    checks = [
        ("metrics.db",      MetricsSessionLocal),
        ("sensor_data.db",  SensorSessionLocal),
    ]
    for db_name, SessionLocal in checks:
        try:
            with SessionLocal() as session:
                session.execute(text("SELECT 1"))
            logger.info("DB health check OK: %s", db_name)
        except Exception as exc:
            raise RuntimeError(f"Database health check FAILED for {db_name}: {exc}") from exc


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Run startup checks, then yield control to the app, then clean up."""
    logger.info("Starting MP-QUIC dashboard backend…")
    _check_db_connections()
    push_task = asyncio.create_task(ws_broadcaster.start_push_loop())
    logger.info("WebSocket push loop started.")
    yield
    push_task.cancel()
    logger.info("Backend shutdown complete.")


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="MP-QUIC LSTM Dashboard API",
        description=(
            "REST + WebSocket backend for the MP-QUIC network quality prediction dashboard. "
            "Exposes real-time metrics, LSTM prediction results, and degradation scenario state."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    # Allow the Vite dev server (typically :5173) and production build origins.
    allowed_origins = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173",
    ).split(",")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Route registration ────────────────────────────────────────────────────
    app.include_router(metrics_router.router)        # /metrics
    app.include_router(predictions_router.router)    # /predictions
    app.include_router(scenarios_router.router)      # /scenarios
    app.include_router(ws_broadcaster.router)        # /ws  (WebSocket)

    # ── Health endpoint ───────────────────────────────────────────────────────
    @app.get("/health", tags=["system"])
    def health() -> dict:
        """Lightweight liveness probe — no DB query required."""
        return {"status": "ok", "message": "MP-QUIC dashboard backend is running"}

    # ── WebSocket Endpoint ────────────────────────────────────────────────────
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: fastapi.WebSocket) -> None:
        """Accept a WebSocket connection and hold it open until the client disconnects."""
        await websocket.accept()
        await ws_broadcaster._add(websocket)
        # Send an initial handshake confirming the connection is live.
        import json
        await websocket.send_text(json.dumps({"type": "connected", "message": "MP-QUIC dashboard stream active"}))
        try:
            while True:
                # Keep the connection alive; the server pushes data via broadcast().
                await websocket.receive_text()
        except fastapi.WebSocketDisconnect:
            pass
        finally:
            await ws_broadcaster._remove(websocket)

    return app


app = create_app()
