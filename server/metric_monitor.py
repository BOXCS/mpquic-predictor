"""
server/metric_monitor.py — Real-time Metric Monitoring

Computes RTT and goodput per path based on incoming payloads from the
MP-QUIC server. Maintains a sliding window of recent measurements to be
used by the LSTM predictor service.

Does not write to the database or run inference.
"""

import json
import os
import pickle
from collections import deque
from typing import Dict, Any, List

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_SAVED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "saved", "rtt only")
CONFIG_PKL_PATH = os.path.join(MODEL_SAVED_DIR, "config.pkl")

def load_window_size() -> int:
    """Load the sliding window size from the model's config.pkl."""
    try:
        with open(CONFIG_PKL_PATH, "rb") as f:
            config = pickle.load(f)
            return config.get("window_size", 20)
    except FileNotFoundError:
        # Fallback if config.pkl is missing during early dev/tests
        return 20
    except Exception as e:
        print(f"Warning: could not load config.pkl: {e}")
        return 20

# ── Metric Monitor ─────────────────────────────────────────────────────────────
class MetricMonitor:
    def __init__(self):
        self.window_size = load_window_size()
        
        # Maintain a sliding window (deque) for each path
        # 1 = wlan0, 2 = eth0
        self.history: Dict[int, deque] = {
            1: deque(maxlen=self.window_size),
            2: deque(maxlen=self.window_size),
        }

    def process_payload(self, payload: Dict[str, Any], client_ip: str = "unknown") -> None:
        """
        Process an incoming JSON payload from the MP-QUIC server, compute
        goodput, and add the metrics to the sliding window.
        """
        path_id = payload.get("path_id")
        if path_id not in self.history:
            return  # Unknown path
            
        rtt_ms = payload.get("rtt_ms", 0.0)
        
        # Estimate goodput in bps: (payload_size_bits) / (rtt_sec)
        # Using a simplistic calculation based on the raw payload size.
        payload_bytes = len(json.dumps(payload).encode("utf-8"))
        
        if rtt_ms > 0:
            goodput_bps = (payload_bytes * 8) / (rtt_ms / 1000.0)
        else:
            goodput_bps = 0.0

        # Create the metric record
        record = {
            "path_id": path_id,
            "client_ip": client_ip,
            "temperature": payload.get("temperature"),
            "humidity": payload.get("humidity"),
            "rtt_ms": rtt_ms,
            "loss_pct": payload.get("loss_pct", 0.0),
            "goodput_bps": round(goodput_bps, 2)
        }
        
        self.history[path_id].append(record)

        # Write to databases
        from server.db_writer import write_metrics, write_sensor_data
        
        write_metrics(
            path_id=path_id,
            rtt_ms=rtt_ms,
            goodput_bps=round(goodput_bps, 2),
            loss_pct=payload.get("loss_pct", 0.0)
        )
        
        if payload.get("temperature") is not None and payload.get("humidity") is not None:
            write_sensor_data(
                temperature=payload.get("temperature"),
                humidity=payload.get("humidity")
            )

    def get_latest_metrics(self) -> Dict[int, List[Dict[str, Any]]]:
        """
        Expose the current sliding window of metrics for all paths.
        Used by the predictor_service for LSTM inference.
        """
        return {
            1: list(self.history[1]),
            2: list(self.history[2])
        }

# Global singleton instance to be used by mpquic_server.py
monitor = MetricMonitor()
