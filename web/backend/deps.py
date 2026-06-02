"""
web/backend/deps.py — FastAPI Dependency Injection Helpers

Provides SQLAlchemy session generators for use as FastAPI dependencies.
Route handlers call `Depends(get_metrics_db)` or `Depends(get_sensor_db)`;
they never instantiate sessions directly.
"""

from __future__ import annotations

from typing import Generator
from sqlalchemy.orm import Session

from server.db_writer import MetricsSessionLocal, SensorSessionLocal


def get_metrics_db() -> Generator[Session, None, None]:
    """Yield a SQLAlchemy session bound to metrics.db, then close it."""
    db = MetricsSessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_sensor_db() -> Generator[Session, None, None]:
    """Yield a SQLAlchemy session bound to sensor_data.db, then close it."""
    db = SensorSessionLocal()
    try:
        yield db
    finally:
        db.close()
