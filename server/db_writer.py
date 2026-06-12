"""
server/db_writer.py — Centralized Database Writer

Handles all database writes for the MP-QUIC system using SQLAlchemy ORM.
This is the only component permitted to write to the SQLite databases.

Databases:
- data/metrics.db: RTT, goodput, predictions, switching events.
- data/sensor_data.db: Temperature and humidity readings.
"""

import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv

load_dotenv()

# ── Database path resolution ───────────────────────────────────────────────────
# DB_PATH sets the directory that holds both SQLite database files.
#
# Why this exists: when the server runs on Windows (Python 3.11 native) but
# the project root lives on the WSL filesystem (\\wsl$\...), SQLite file
# locking fails with "database is locked" because Windows and WSL use
# different locking primitives on cross-OS mounts.
#
# Solution: point DB_PATH at a Windows-native directory (e.g. C:/mpquic-data/)
# so both the server process and SQLite operate entirely within the Windows
# NTFS filesystem, avoiding the cross-OS locking issue entirely.
#
# On Linux / WSL-only deployments, leave DB_PATH unset and the default
# project-relative data/ directory is used (existing behaviour).
_DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DATA_DIR: str = os.getenv("DB_PATH", _DEFAULT_DATA_DIR).rstrip("/\\")

os.makedirs(DATA_DIR, exist_ok=True)

METRICS_DB_PATH: str = os.path.join(DATA_DIR, "metrics.db")
SENSOR_DB_PATH: str  = os.path.join(DATA_DIR, "sensor_data.db")

metrics_engine = create_engine(f"sqlite:///{METRICS_DB_PATH}", future=True)
sensor_engine = create_engine(f"sqlite:///{SENSOR_DB_PATH}", future=True)

MetricsBase = declarative_base()
SensorBase = declarative_base()

# ── ORM Models: metrics.db ────────────────────────────────────────────────────
class NetworkMetric(MetricsBase):
    __tablename__ = "network_metrics"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    path_id = Column(Integer, nullable=False)
    rtt_ms = Column(Float, nullable=False)
    goodput_bps = Column(Float, nullable=False)
    loss_pct = Column(Float, default=0.0)

class PredictionResult(MetricsBase):
    __tablename__ = "prediction_results"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    predicted_path = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=False)
    degradation_detected = Column(Boolean, nullable=False)

class SwitchingEvent(MetricsBase):
    __tablename__ = "switching_events"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    from_path = Column(Integer, nullable=False)
    to_path = Column(Integer, nullable=False)
    reason = Column(String, nullable=False)

# ── ORM Models: sensor_data.db ────────────────────────────────────────────────
class SensorReading(SensorBase):
    __tablename__ = "sensor_readings"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    temperature = Column(Float, nullable=False)
    humidity = Column(Float, nullable=False)

# ── Create Tables ──────────────────────────────────────────────────────────────
MetricsBase.metadata.create_all(metrics_engine)
SensorBase.metadata.create_all(sensor_engine)

MetricsSessionLocal = sessionmaker(bind=metrics_engine, future=True)
SensorSessionLocal = sessionmaker(bind=sensor_engine, future=True)

# ── Public Functions ───────────────────────────────────────────────────────────
def write_metrics(path_id: int, rtt_ms: float, goodput_bps: float, loss_pct: float = 0.0) -> None:
    """Write RTT, goodput, and loss metrics to metrics.db."""
    with MetricsSessionLocal() as session:
        record = NetworkMetric(
            path_id=path_id,
            rtt_ms=rtt_ms,
            goodput_bps=goodput_bps,
            loss_pct=loss_pct
        )
        session.add(record)
        session.commit()

def write_prediction(predicted_path: int, confidence: float, degradation_detected: bool) -> None:
    """Write LSTM inference prediction results to metrics.db."""
    with MetricsSessionLocal() as session:
        record = PredictionResult(
            predicted_path=predicted_path,
            confidence=confidence,
            degradation_detected=degradation_detected
        )
        session.add(record)
        session.commit()

def write_switching_event(from_path: int, to_path: int, reason: str) -> None:
    """Write path switching events to metrics.db."""
    with MetricsSessionLocal() as session:
        record = SwitchingEvent(
            from_path=from_path,
            to_path=to_path,
            reason=reason
        )
        session.add(record)
        session.commit()

def write_sensor_data(temperature: float, humidity: float) -> None:
    """Write temperature and humidity readings to sensor_data.db."""
    with SensorSessionLocal() as session:
        record = SensorReading(
            temperature=temperature,
            humidity=humidity
        )
        session.add(record)
        session.commit()
