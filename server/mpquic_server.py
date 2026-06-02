"""
server/mpquic_server.py — MP-QUIC Receiver for IoT Sensor Data

Receives telemetry and network metrics from the Raspberry Pi over two
simultaneous QUIC connections (wlan0 + eth0). Parses the JSON payloads
and forwards them to the metric_monitor for RTT/goodput calculation.

Inference and database logic are excluded from this module.
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
from typing import Optional

# Ensure project root is in sys.path when running as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from aioquic.asyncio import serve
from aioquic.quic.configuration import QuicConfiguration
from aioquic.asyncio.protocol import QuicConnectionProtocol
from aioquic.quic.events import StreamDataReceived

# ── Setup Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("mpquic_server")

# ── Load Configuration ─────────────────────────────────────────────────────────
load_dotenv()

SERVER_HOST = os.getenv("SERVER_HOST", "127.0.0.1")
SERVER_PORT_PATH1 = int(os.getenv("SERVER_PORT_PATH1", "5001"))
SERVER_PORT_PATH2 = int(os.getenv("SERVER_PORT_PATH2", "5002"))

_raw_ips = os.getenv("ALLOWED_CLIENT_IPS", "127.0.0.1")
ALLOWED_CLIENT_IPS = [ip.strip() for ip in _raw_ips.split(",") if ip.strip()]

# ── TLS Certificates ───────────────────────────────────────────────────────────
CERT_FILE = os.path.join(os.path.dirname(__file__), "server.crt")
KEY_FILE = os.path.join(os.path.dirname(__file__), "server.key")

def ensure_tls_certificates():
    """Generate self-signed certificates for local QUIC development if missing."""
    if os.path.exists(CERT_FILE) and os.path.exists(KEY_FILE):
        return

    logger.info("Generating self-signed TLS certificates for aioquic...")
    try:
        subprocess.run(
            [
                "openssl", "req", "-x509", "-newkey", "rsa:2048",
                "-keyout", KEY_FILE, "-out", CERT_FILE,
                "-days", "365", "-nodes", "-subj", "/CN=localhost"
            ],
            check=True,
            capture_output=True
        )
        logger.info(f"Certificates generated: {CERT_FILE}, {KEY_FILE}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to generate certificates: {e.stderr.decode()}")
        sys.exit(1)
    except FileNotFoundError:
        logger.error("openssl is not installed. Please install it to generate certificates.")
        sys.exit(1)

# ── Metric Monitor ─────────────────────────────────────────────────────────────
from server.metric_monitor import monitor as metric_monitor

# ── Active Connections Registry ────────────────────────────────────────────────
# Used by predictor_service to broadcast switching recommendations.
active_connections = set()

# ── QUIC Protocol ──────────────────────────────────────────────────────────────
class MPQuicServerProtocol(QuicConnectionProtocol):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._client_ip: Optional[str] = None
        self._buffer = b""

    def connection_made(self, transport):
        super().connection_made(transport)
        peername = transport.get_extra_info("peername")
        if peername:
            self._client_ip = peername[0]
            
            # ── IP Allowlist Verification ──
            if self._client_ip not in ALLOWED_CLIENT_IPS:
                logger.warning(f"Rejected connection from unauthorized IP: {self._client_ip}")
                self.close()
                return

            logger.debug(f"Accepted connection from authorized IP: {self._client_ip}")
        else:
            logger.debug("Accepted connection (peername not available)")
            
        active_connections.add(self)

    def connection_lost(self, exc):
        active_connections.discard(self)
        super().connection_lost(exc)

    def quic_event_received(self, event):
        if isinstance(event, StreamDataReceived):
            self._buffer += event.data
            
            # Simple line-based JSON framing: assumes each JSON object is followed by a newline,
            # or we just try to parse the whole buffer if it's sent in chunks.
            # network_emulator will send one JSON payload and optionally close the stream.
            
            try:
                payload = json.loads(self._buffer.decode("utf-8"))
                self._buffer = b"" # clear buffer after successful parse
                metric_monitor.process_payload(payload, self._client_ip)
            except json.JSONDecodeError:
                # Buffer might be incomplete, wait for more data
                pass

# ── Server Boot ────────────────────────────────────────────────────────────────
async def run_server():
    from server.predictor_service import start_predictor_service
    ensure_tls_certificates()

    configuration = QuicConfiguration(is_client=False)
    configuration.load_cert_chain(CERT_FILE, KEY_FILE)
    
    # Start predictor loop in background
    asyncio.create_task(start_predictor_service())

    logger.info(f"Starting MP-QUIC server...")
    logger.info(f"Allowed Client IPs: {ALLOWED_CLIENT_IPS}")
    logger.info(f"Path 1 Listener (wlan0) on UDP {SERVER_HOST}:{SERVER_PORT_PATH1}")
    logger.info(f"Path 2 Listener (eth0)  on UDP {SERVER_HOST}:{SERVER_PORT_PATH2}")

    # Start two listeners concurrently
    try:
        await asyncio.gather(
            serve(
                host=SERVER_HOST,
                port=SERVER_PORT_PATH1,
                configuration=configuration,
                create_protocol=MPQuicServerProtocol,
            ),
            serve(
                host=SERVER_HOST,
                port=SERVER_PORT_PATH2,
                configuration=configuration,
                create_protocol=MPQuicServerProtocol,
            )
        )
        # Keep the server running
        await asyncio.Future()
    except asyncio.CancelledError:
        logger.info("Server shut down.")

if __name__ == "__main__":
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        pass
