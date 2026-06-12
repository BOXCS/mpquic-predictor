"""
hardware/mpquic_client.py — Dual-path MP-QUIC client for the Raspberry Pi.

Runs on the Raspberry Pi only.  Must not be imported on the server side.

This module:
  1. Opens a persistent QUIC connection to the server on each configured path.
  2. At every SEND_INTERVAL_SEC tick, reads the DHT11 sensor via
     sensor_reader.read_sensor() and sends a JSON telemetry payload on
     all active paths concurrently (asyncio.gather).
  3. Listens on each connection for incoming QUIC stream data (switching
     recommendations pushed by predictor_service.py on the server) and
     forwards them to path_switcher.handle_recommendation().

Path availability
─────────────────
Path 1 (wlan0 / WiFi)   — always active.
Path 2 (eth0 / LAN)     — activated only when ENABLE_PATH2=true in .env or
                           via --enable-path2 CLI flag.  Graceful degradation:
                           if Path 2 cannot connect, the client continues on
                           Path 1 alone and logs a warning.

Payload schema (matches what server/metric_monitor.py expects)
──────────────────────────────────────────────────────────────
{
    "path_id":     int,    # 1 = wlan0, 2 = eth0
    "temperature": float,  # °C from DHT11, or null on sensor failure
    "humidity":    float,  # %RH from DHT11, or null on sensor failure
    "rtt_ms":      float,  # round-trip time measured by this client (ms)
    "loss_pct":    float,  # always 0.0 — real hardware doesn't simulate loss
    "timestamp_ms": int,   # Unix epoch milliseconds at send time
}

Usage (on the Raspberry Pi)
────────────────────────────
    # Path 1 only (default):
    python3 -m hardware.mpquic_client

    # Both paths (when LAN cable is connected):
    python3 -m hardware.mpquic_client --enable-path2

Configuration (all sourced from hardware/config.py → .env)
────────────────────────────────────────────────────────────
    SERVER_HOST         — IP of the server to connect to
    SERVER_PORT_PATH1   — UDP port for the wlan0 QUIC listener
    SERVER_PORT_PATH2   — UDP port for the eth0 QUIC listener
    SEND_INTERVAL_SEC   — seconds between sensor transmissions per path
    PATH1_IFACE         — interface name for path 1 (wlan0)
    PATH2_IFACE         — interface name for path 2 (eth0)
"""

import argparse
import asyncio
import json
import logging
import os
import ssl
import sys
import time
from typing import Optional

from aioquic.asyncio import connect
from aioquic.quic.configuration import QuicConfiguration
from aioquic.asyncio.protocol import QuicConnectionProtocol
from aioquic.quic.events import StreamDataReceived
from dotenv import load_dotenv

# ── Project root on sys.path (needed when run as a script) ───────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

load_dotenv()

from hardware.config import (
    SERVER_HOST,
    SERVER_PORT_PATH1,
    SERVER_PORT_PATH2,
    SEND_INTERVAL_SEC,
    PATH1_ID,
    PATH2_ID,
    PATH1_IFACE,
    PATH2_IFACE,
)
from hardware.sensor_reader import read_sensor

# path_switcher is imported lazily inside the recommendation handler so that
# this module can be tested without path_switcher being present yet.

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("mpquic_client")

# ── Path routing table ────────────────────────────────────────────────────────
# Maps path_id → (server_host, server_port, interface_label)
_PATH_CONFIG: dict[int, dict] = {
    PATH1_ID: {
        "host":  SERVER_HOST,
        "port":  SERVER_PORT_PATH1,
        "iface": PATH1_IFACE,
        "label": f"Path 1 ({PATH1_IFACE}/WiFi)",
    },
    PATH2_ID: {
        "host":  SERVER_HOST,
        "port":  SERVER_PORT_PATH2,
        "iface": PATH2_IFACE,
        "label": f"Path 2 ({PATH2_IFACE}/LAN)",
    },
}


# ── QUIC protocol — client side ───────────────────────────────────────────────

class MPQuicClientProtocol(QuicConnectionProtocol):
    """
    Persistent QUIC connection that:
      - sends telemetry on open QUIC streams.
      - receives switching recommendations pushed by the server.
    """

    def __init__(self, path_id: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._path_id = path_id
        self._recv_buffer = b""

    def quic_event_received(self, event) -> None:
        """Handle incoming events — server-push switching recommendations."""
        if isinstance(event, StreamDataReceived):
            self._recv_buffer += event.data
            # Try to decode complete newline-terminated JSON messages.
            while b"\n" in self._recv_buffer:
                line, self._recv_buffer = self._recv_buffer.split(b"\n", 1)
                if line.strip():
                    self._handle_server_recommendation(line)

    def _handle_server_recommendation(self, raw: bytes) -> None:
        """Parse and forward a switching recommendation from the server."""
        try:
            recommendation: dict = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.warning(
                "[%s] Received malformed server push (ignored): %s",
                _PATH_CONFIG[self._path_id]["label"],
                exc,
            )
            return

        logger.info(
            "[%s] Switching recommendation received: %s",
            _PATH_CONFIG[self._path_id]["label"],
            recommendation,
        )

        # Forward to path_switcher — imported lazily to allow partial testing.
        try:
            from hardware import path_switcher  # noqa: PLC0415
            path_switcher.handle_recommendation(recommendation)
        except ImportError:
            logger.debug("path_switcher not yet available — recommendation logged only.")
        except Exception as exc:
            logger.error("path_switcher.handle_recommendation raised: %s", exc)


# ── Payload builder ───────────────────────────────────────────────────────────

def _build_payload(
    path_id: int,
    temperature_c: float | None,
    humidity_pct: float | None,
    rtt_ms: float,
) -> bytes:
    """Serialise a telemetry payload to a newline-terminated UTF-8 JSON byte string."""
    payload: dict = {
        "path_id":      path_id,
        "temperature":  temperature_c,
        "humidity":     humidity_pct,
        "rtt_ms":       rtt_ms,
        "loss_pct":     0.0,   # real hardware does not simulate loss
        "timestamp_ms": int(time.time() * 1000),
    }
    return (json.dumps(payload) + "\n").encode("utf-8")


# ── Per-path send ─────────────────────────────────────────────────────────────

async def _send_on_path(
    path_id: int,
    client: MPQuicClientProtocol,
    temperature_c: float | None,
    humidity_pct: float | None,
) -> dict:
    """
    Send one telemetry payload on the given path and return a result dict.

    RTT is measured as the wall-clock time between queuing the stream data
    and calling transmit().  This is an application-level RTT proxy — the
    true QUIC RTT reported by the stack is not exposed by aioquic's public API.
    """
    label = _PATH_CONFIG[path_id]["label"]
    try:
        stream_id = client._quic.get_next_available_stream_id()
        t0 = time.monotonic()

        raw_payload = _build_payload(path_id, temperature_c, humidity_pct, rtt_ms=0.0)

        client._quic.send_stream_data(stream_id, raw_payload, end_stream=True)
        client.transmit()

        rtt_ms = round((time.monotonic() - t0) * 1000.0, 2)

        # Re-send the payload with the measured RTT so the server can compute goodput.
        # Open a new stream for the corrected payload.
        corrected_stream_id = client._quic.get_next_available_stream_id()
        corrected_payload = _build_payload(path_id, temperature_c, humidity_pct, rtt_ms)
        client._quic.send_stream_data(corrected_stream_id, corrected_payload, end_stream=True)
        client.transmit()

        logger.info(
            "[%s] sent: temp=%.1f°C  hum=%.1f%%RH  rtt_ms=%.1f",
            label,
            temperature_c if temperature_c is not None else float("nan"),
            humidity_pct if humidity_pct is not None else float("nan"),
            rtt_ms,
        )
        return {"path_id": path_id, "rtt_ms": rtt_ms, "status": "ok"}

    except Exception as exc:
        logger.error("[%s] send failed: %s", label, exc)
        return {"path_id": path_id, "rtt_ms": 0.0, "status": f"error:{type(exc).__name__}"}


# ── Main client loop ──────────────────────────────────────────────────────────

async def _run_path(
    path_id: int,
    active_paths: list[int],
    send_interval_sec: float,
) -> None:
    """
    Connect to the server on one path and loop forever, sending telemetry
    and processing incoming recommendations.

    If the connection fails on startup, a warning is logged and this coroutine
    returns — the caller (run_client) continues running the other paths.
    """
    cfg = _PATH_CONFIG[path_id]
    quic_config = QuicConfiguration(is_client=True, verify_mode=ssl.CERT_NONE)

    logger.info(
        "Connecting on %s → %s:%d (QUIC UDP)",
        cfg["label"],
        cfg["host"],
        cfg["port"],
    )

    try:
        async with connect(
            cfg["host"],
            cfg["port"],
            configuration=quic_config,
            create_protocol=lambda *a, **kw: MPQuicClientProtocol(path_id, *a, **kw),
        ) as client:
            logger.info("[%s] connection established.", cfg["label"])

            while True:
                # Read sensor — non-blocking, returns (None, None) on failure.
                temperature_c, humidity_pct = read_sensor()

                if temperature_c is None or humidity_pct is None:
                    logger.warning(
                        "[%s] Sensor read returned None — sending null values this tick.",
                        cfg["label"],
                    )

                await _send_on_path(path_id, client, temperature_c, humidity_pct)
                await asyncio.sleep(send_interval_sec)

    except ConnectionRefusedError:
        logger.warning(
            "[%s] Connection refused at %s:%d. "
            "Is mpquic_server.py running on the server?",
            cfg["label"],
            cfg["host"],
            cfg["port"],
        )
    except OSError as exc:
        logger.warning(
            "[%s] OS error connecting (%s). "
            "Check that %s is up and SERVER_HOST is correct.",
            cfg["label"],
            exc,
            cfg["iface"],
        )
    except Exception as exc:
        logger.error("[%s] Unexpected error: %s", cfg["label"], exc)


async def run_client(
    enable_path2: bool = False,
    send_interval_sec: float = SEND_INTERVAL_SEC,
) -> None:
    """
    Launch all active path coroutines concurrently.

    Path 1 (wlan0) is always included.
    Path 2 (eth0) is included only when enable_path2=True.
    If Path 2 fails to connect, Path 1 continues unaffected.
    """
    active_paths = [PATH1_ID]
    if enable_path2:
        active_paths.append(PATH2_ID)

    logger.info(
        "MP-QUIC client starting | paths=%s | interval=%.2fs",
        active_paths,
        send_interval_sec,
    )

    path_coroutines = [
        _run_path(pid, active_paths, send_interval_sec)
        for pid in active_paths
    ]

    # gather(return_exceptions=True) so one path failure doesn't kill the other.
    results = await asyncio.gather(*path_coroutines, return_exceptions=True)

    for path_id, result in zip(active_paths, results):
        if isinstance(result, Exception):
            logger.error(
                "[%s] Path coroutine raised uncaught exception: %s",
                _PATH_CONFIG[path_id]["label"],
                result,
            )


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MP-QUIC hardware client — sends DHT11 telemetry to the server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 -m hardware.mpquic_client\n"
            "  python3 -m hardware.mpquic_client --enable-path2\n"
            "  python3 -m hardware.mpquic_client --interval 1.0\n"
        ),
    )
    parser.add_argument(
        "--enable-path2",
        action="store_true",
        default=os.getenv("ENABLE_PATH2", "false").lower() == "true",
        help=(
            "Activate Path 2 (eth0/LAN). "
            "Also set via ENABLE_PATH2=true in .env. "
            "Default: disabled (Path 1 / wlan0 only)."
        ),
    )
    parser.add_argument(
        "--interval",
        metavar="SEC",
        type=float,
        default=SEND_INTERVAL_SEC,
        help=f"Send interval in seconds per path (default: {SEND_INTERVAL_SEC}, from SEND_INTERVAL_SEC env).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        asyncio.run(run_client(enable_path2=args.enable_path2, send_interval_sec=args.interval))
    except KeyboardInterrupt:
        logger.info("mpquic_client: stopped by user.")


if __name__ == "__main__":
    main()
