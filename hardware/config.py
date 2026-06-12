"""
hardware/config.py — Central configuration for all hardware-layer components.

All IP addresses, ports, intervals, and thresholds are defined here.
Components must never hardcode these values inline; import from this module.
All values are sourced from environment variables so the same codebase runs
on both the dev machine (with simulator defaults) and the Raspberry Pi
(with LAN hardware values).

Environment variable overrides (set in .env or shell export):

  Network / Server connectivity
  ─────────────────────────────
  SERVER_HOST          — LAN IP of the server the Pi connects to
                         Dev default : 127.0.0.1
                         Hardware    : 192.168.1.10

  SERVER_PORT_PATH1    — UDP port for the wlan0 QUIC listener (path 1)
                         Default: 5001

  SERVER_PORT_PATH2    — UDP port for the eth0 QUIC listener (path 2)
                         Default: 5002

  ALLOWED_CLIENT_IPS   — Comma-separated list of client IPs the server accepts.
                         Dev default : 127.0.0.1
                         Hardware    : 192.168.1.18  (Raspberry Pi LAN IP)

  Timing
  ──────
  SEND_INTERVAL_SEC    — Seconds between sensor/emulator transmissions per path
                         Default: 0.5

  SENSOR_INTERVAL_SEC  — DHT11 read interval in seconds on the Raspberry Pi
                         Default: 2.0

  INFERENCE_INTERVAL_SEC — Seconds between LSTM inference cycles on the server
                           Default: 1.0

  Thresholds
  ──────────
  RTT_DEGRADATION_MS   — RTT (ms) above which a path is flagged as degraded.
                         Used for feature engineering (status_enc) in predictor.
                         Default: 100.0

  Path switching hysteresis
  ─────────────────────────
  SWITCH_MARGIN_PCT    — Alt path avg RTT must be at least this % lower than the
                         active path before a switch is triggered.
                         Default: 20

  SWITCH_COOLDOWN_SEC  — Minimum seconds between two consecutive switches.
                         Default: 30

  Hardware (Raspberry Pi)
  ────────────────────────
  DHT_GPIO_PIN         — BCM GPIO pin number the DHT11 data line is wired to.
                         Hardware: 4  (BCM pin 4 with built-in pull-up on module)

  DHT_SENSOR_TYPE      — Sensor variant: "DHT11" or "DHT22".
                         Hardware: DHT11

  PATH1_IFACE          — Network interface name for path 1 (WiFi).
                         Default: wlan0

  PATH2_IFACE          — Network interface name for path 2 (Ethernet/LAN).
                         Default: eth0

  PATH_SWITCH_MECHANISM — How the client executes a path switch.
                          "socket_bind" — application-level socket binding
                                          (no sudo required; current choice).
                          "ip_rule"     — OS-level routing rule (requires sudo).
                          Default: socket_bind
"""

import os

# ── Server connectivity ────────────────────────────────────────────────────────
SERVER_HOST: str = os.getenv("SERVER_HOST", "127.0.0.1")
SERVER_PORT_PATH1: int = int(os.getenv("SERVER_PORT_PATH1", "5001"))
SERVER_PORT_PATH2: int = int(os.getenv("SERVER_PORT_PATH2", "5002"))

# Comma-separated list; loaded as a Python list for use in mpquic_server.py.
_raw_allowed = os.getenv("ALLOWED_CLIENT_IPS", "127.0.0.1")
ALLOWED_CLIENT_IPS: list[str] = [ip.strip() for ip in _raw_allowed.split(",") if ip.strip()]

# ── Timing ────────────────────────────────────────────────────────────────────
# How often the client (or emulator) sends a packet on each path.
SEND_INTERVAL_SEC: float = float(os.getenv("SEND_INTERVAL_SEC", "0.5"))

# How often the DHT11 sensor is read on the Raspberry Pi.
SENSOR_INTERVAL_SEC: float = float(os.getenv("SENSOR_INTERVAL_SEC", "2.0"))

# How often the predictor service runs inference on the server.
INFERENCE_INTERVAL_SEC: float = float(os.getenv("INFERENCE_INTERVAL_SEC", "1.0"))

# ── Degradation thresholds ────────────────────────────────────────────────────
# RTT above this value marks a path as degraded (used in feature engineering).
RTT_DEGRADATION_MS: float = float(os.getenv("RTT_DEGRADATION_MS", "100.0"))

# ── Path switching hysteresis ─────────────────────────────────────────────────
# Read by predictor_service.py. Defined here so hardware/ and server/ share one
# source of truth without a circular import.
SWITCH_MARGIN_PCT: float = float(os.getenv("SWITCH_MARGIN_PCT", "20"))
SWITCH_COOLDOWN_SEC: float = float(os.getenv("SWITCH_COOLDOWN_SEC", "30"))

# ── Hardware — sensor ─────────────────────────────────────────────────────────
# BCM GPIO pin the DHT11 data line is connected to (built-in pull-up on module).
DHT_GPIO_PIN: int = int(os.getenv("DHT_GPIO_PIN", "4"))

# Sensor variant: "DHT11" or "DHT22". Affects adafruit_dht initialisation.
DHT_SENSOR_TYPE: str = os.getenv("DHT_SENSOR_TYPE", "DHT11")

# ── Hardware — network interfaces ────────────────────────────────────────────
PATH1_IFACE: str = os.getenv("PATH1_IFACE", "wlan0")   # WiFi (path 1)
PATH2_IFACE: str = os.getenv("PATH2_IFACE", "eth0")    # Ethernet/LAN (path 2)

# ── Hardware — path switching mechanism ───────────────────────────────────────
# "socket_bind" — mpquic_client.py binds the sending socket to the specific
#                  interface address at the application layer (no sudo needed).
# "ip_rule"     — uses OS-level 'ip rule' / 'ip route' commands (requires sudo).
PATH_SWITCH_MECHANISM: str = os.getenv("PATH_SWITCH_MECHANISM", "socket_bind")

# ── Path identifiers ──────────────────────────────────────────────────────────
PATH1_ID: int = 1   # wlan0 / WiFi
PATH2_ID: int = 2   # eth0  / Ethernet

# ── Derived endpoint addresses ─────────────────────────────────────────────────
# Used by the QUIC client to know which (host, port) to connect to per path.
def path1_addr() -> tuple[str, int]:
    """Return (host, port) for the wlan0 QUIC path."""
    return SERVER_HOST, SERVER_PORT_PATH1


def path2_addr() -> tuple[str, int]:
    """Return (host, port) for the eth0 QUIC path."""
    return SERVER_HOST, SERVER_PORT_PATH2


# Legacy URL helpers kept for backward compatibility with simulator code.
def path1_url() -> str:
    return f"http://{SERVER_HOST}:{SERVER_PORT_PATH1}/path1"


def path2_url() -> str:
    return f"http://{SERVER_HOST}:{SERVER_PORT_PATH2}/path2"
