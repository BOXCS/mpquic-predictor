"""
hardware/config.py — Central configuration for all hardware-layer components.

All IP addresses, ports, intervals, and thresholds are defined here.
Components must never hardcode these values inline; import from this module
or read from the environment variables listed below.

Environment variable overrides (via .env or shell export):
    SERVER_HOST          — IP of the MP-QUIC / HTTP server (default: 127.0.0.1)
    SERVER_PORT_PATH1    — Port for path 1 endpoint (default: 5001)
    SERVER_PORT_PATH2    — Port for path 2 endpoint (default: 5002)
    SEND_INTERVAL_SEC    — Seconds between sensor/emulator transmissions (default: 0.5)
    RTT_DEGRADATION_MS   — RTT threshold (ms) above which a path is considered degraded (default: 100)
    SENSOR_INTERVAL_SEC  — DHT22 read interval in seconds (default: 2.0)
    PATH1_IFACE          — Network interface for path 1 (default: wlan0)
    PATH2_IFACE          — Network interface for path 2 (default: eth0)
"""

import os

# ── Server connectivity ────────────────────────────────────────────────────────
SERVER_HOST: str   = os.getenv("SERVER_HOST", "127.0.0.1")
SERVER_PORT_PATH1: int = int(os.getenv("SERVER_PORT_PATH1", "5001"))
SERVER_PORT_PATH2: int = int(os.getenv("SERVER_PORT_PATH2", "5002"))

# ── Timing ────────────────────────────────────────────────────────────────────
# How often the client (or emulator) sends a packet on each path.
SEND_INTERVAL_SEC: float = float(os.getenv("SEND_INTERVAL_SEC", "0.5"))

# How often the DHT22 sensor is read on the Raspberry Pi.
SENSOR_INTERVAL_SEC: float = float(os.getenv("SENSOR_INTERVAL_SEC", "2.0"))

# How often the predictor service runs inference on the server.
INFERENCE_INTERVAL_SEC: float = float(os.getenv("INFERENCE_INTERVAL_SEC", "1.0"))

# ── Degradation thresholds ────────────────────────────────────────────────────
# RTT above this value triggers a switching recommendation from the server.
RTT_DEGRADATION_MS: float = float(os.getenv("RTT_DEGRADATION_MS", "100.0"))

# ── Network interfaces (Raspberry Pi) ─────────────────────────────────────────
PATH1_IFACE: str = os.getenv("PATH1_IFACE", "wlan0")   # WiFi
PATH2_IFACE: str = os.getenv("PATH2_IFACE", "eth0")    # LAN

# ── Path identifiers ──────────────────────────────────────────────────────────
PATH1_ID: int = 1   # wlan0 / WiFi
PATH2_ID: int = 2   # eth0  / LAN

# ── Endpoint URLs ─────────────────────────────────────────────────────────────
def path1_url() -> str:
    return f"http://{SERVER_HOST}:{SERVER_PORT_PATH1}/path1"

def path2_url() -> str:
    return f"http://{SERVER_HOST}:{SERVER_PORT_PATH2}/path2"
