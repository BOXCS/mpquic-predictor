"""
simulator/network_emulator.py — Hardware-absent MP-QUIC traffic emulator.

Acts as a drop-in substitute for the Raspberry Pi during development and
lab phases when physical hardware is unavailable.  It replays the 14
degradation scenarios defined in simulator/data_generator.py, sending
simulated metric payloads to the server concurrently over two logical paths
(path 1 ≡ wlan0/WiFi, path 2 ≡ eth0/LAN) using asyncio.

Transport note
--------------
The target architecture uses aioquic for actual MP-QUIC transport on the
Raspberry Pi.  This emulator runs on a development machine where aioquic is
not installed; it therefore uses asyncio + requests (via asyncio.to_thread)
to send concurrent HTTP POST requests to the server stubs at ports 5001 and
5002.  The payload schema and log format are identical to what the real
hardware client will produce, so the server and LSTM pipeline are unaffected
when switching between emulated and real traffic.

Usage
-----
    # Start the server stub first (one terminal):
    python server/server.py

    # Then run the emulator (another terminal):
    python simulator/network_emulator.py

    # Run a single named scenario:
    python simulator/network_emulator.py --scenario normal

    # Run with a custom send interval (overrides config.py / .env):
    python simulator/network_emulator.py --interval 0.25

Protected files (never modified by this module):
    simulator/data_generator.py
    data/logs/path_log.csv   (append-only via logger/path_logger.py)
"""

import asyncio
import argparse
import random
import sys
import os
import time
from typing import Optional
import json
import ssl

from aioquic.asyncio import connect
from aioquic.quic.configuration import QuicConfiguration
from aioquic.asyncio.protocol import QuicConnectionProtocol

# ── Project root on sys.path ───────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Internal imports ───────────────────────────────────────────────────────────
from hardware.config import (
    SEND_INTERVAL_SEC,
    RTT_DEGRADATION_MS,
    PATH1_ID,
    PATH2_ID,
    SERVER_HOST,
    SERVER_PORT_PATH1,
    SERVER_PORT_PATH2,
)
from logger.path_logger import init_logger, log_entry

# Import the scenario list from data_generator WITHOUT running any side effects.
# We import SCENARIOS directly — data_generator's if __name__ == '__main__'
# block is never executed.
from simulator.data_generator import SCENARIOS


# ── Path configuration ────────────────────────────────────────────────────────

_PATH_CONFIG: dict[int, dict] = {
    PATH1_ID: {
        "label": "wlan0 (WiFi)",
        "host": SERVER_HOST,
        "port": SERVER_PORT_PATH1,
    },
    PATH2_ID: {
        "label": "eth0 (LAN)",
        "host": SERVER_HOST,
        "port": SERVER_PORT_PATH2,
    },
}


# ── Metric simulation helpers ─────────────────────────────────────────────────

def _sample_condition(condition: dict, elapsed: float, scenario: dict) -> dict:
    """
    Return a condition dict that reflects any gradual transition defined in
    the scenario.  The original condition dict is not mutated.
    """
    cond = dict(condition)
    transition = scenario.get("transition")
    if not transition:
        return cond

    t_start: float = transition["start_at"]
    t_end: float   = transition["end_at"]
    t_path: int    = transition["path"]

    base_key = f"path{t_path}"
    if base_key not in scenario:
        return cond

    if t_start <= elapsed <= t_end:
        progress = (elapsed - t_start) / (t_end - t_start)
        cond["base_delay"] = (
            scenario[base_key]["base_delay"]
            + progress * (transition["target_delay"] - scenario[base_key]["base_delay"])
        )
        cond["loss"] = (
            scenario[base_key]["loss"]
            + progress * (transition["target_loss"] - scenario[base_key]["loss"])
        )
    elif elapsed > t_end:
        cond["base_delay"] = transition["target_delay"]
        cond["loss"]       = transition["target_loss"]

    return cond


def _compute_rtt(condition: dict) -> Optional[float]:
    """
    Simulate RTT (ms) from a condition dict.
    Returns None when the packet is simulated as dropped.
    """
    loss_pct: float = condition.get("loss", 0.0)
    if random.random() < loss_pct / 100.0:
        return None  # dropped

    base_delay: float  = condition["base_delay"]
    jitter: float      = condition.get("jitter", 0.0)
    delay = base_delay + random.uniform(-jitter, jitter)
    return max(1.0, round(delay, 2))


def _compute_goodput(payload_bytes: int, rtt_ms: float) -> int:
    """
    Approximate goodput in bps from payload size and RTT.
    goodput ≈ (payload_bits) / (rtt_seconds)
    """
    if rtt_ms <= 0:
        return 0
    return int(payload_bytes * 8 / (rtt_ms / 1000.0))


# ── Async per-path sender ─────────────────────────────────────────────────────

async def _send_on_path(
    path_id: int,
    rtt_ms: float,
    condition: dict,
    send_interval_sec: float,
    client: QuicConnectionProtocol,
) -> dict:
    """
    Send one simulated sensor payload to the server on the given path and
    return a result dict with RTT, goodput, and status.

    The simulated delay is applied before the actual HTTP send so that the
    server sees traffic that already reflects the network conditions.  Both
    paths run concurrently via asyncio.gather in the caller.
    """
    loss_pct: float = condition.get("loss", 0.0)

    # --- Dropped packet ---
    if rtt_ms is None:
        log_entry(path_id, 0, 0, round(loss_pct, 2), "dropped")
        return {
            "path_id":         path_id,
            "rtt_ms":          0,
            "goodput_bps":     0,
            "packet_loss_pct": round(loss_pct, 2),
            "status":          "dropped",
        }

    # --- Simulate network delay (non-blocking) ---
    await asyncio.sleep(rtt_ms / 1000.0)

    # --- Build payload (mimics sensor_reader output) ---
    payload = {
        "path_id":     path_id,
        "temperature": round(random.uniform(25.0, 35.0), 1),
        "humidity":    round(random.uniform(60.0, 90.0), 1),
        "rtt_ms":      rtt_ms,             # pre-computed for server reference
        "loss_pct":    round(loss_pct, 2),
    }
    payload_bytes = len(str(payload).encode())

    # --- Send (via persistent aioquic QUIC stream) ---
    try:
        t_send = time.monotonic()
        
        stream_id = client._quic.get_next_available_stream_id()
        payload_str = json.dumps(payload) + "\n"
        client._quic.send_stream_data(stream_id, payload_str.encode("utf-8"), end_stream=True)
        client.transmit()

        wire_rtt = round((time.monotonic() - t_send) * 1000.0, 2)
        effective_rtt = max(rtt_ms, wire_rtt)

        goodput_bps = _compute_goodput(payload_bytes, effective_rtt)
        status      = "success"

        log_entry(path_id, effective_rtt, goodput_bps, round(loss_pct, 2), status)
        return {
            "path_id":         path_id,
            "rtt_ms":          effective_rtt,
            "goodput_bps":     goodput_bps,
            "packet_loss_pct": round(loss_pct, 2),
            "status":          status,
        }

    except Exception as exc:
        log_entry(path_id, 0, 0, round(loss_pct, 2), "error")
        return {
            "path_id":         path_id,
            "rtt_ms":          0,
            "goodput_bps":     0,
            "packet_loss_pct": round(loss_pct, 2),
            "status":          f"error:{type(exc).__name__}",
        }


# ── Scenario runner ────────────────────────────────────────────────────────────

async def run_scenario(
    scenario: dict,
    send_interval_sec: float,
    *,
    verbose: bool = True,
) -> int:
    """
    Replay one scenario asynchronously.  Both paths are sent concurrently
    per iteration via asyncio.gather.

    Returns the total number of records logged (both paths combined).
    """
    name: str             = scenario["name"]
    duration_sec: float   = scenario["duration_seconds"]
    cond1_base: dict      = scenario["path1"]
    cond2_base: dict      = scenario["path2"]

    if verbose:
        print(f"\n[SCENARIO] {name} — {int(duration_sec)}s "
              f"| interval={send_interval_sec}s "
              f"| paths: {_PATH_CONFIG[PATH1_ID]['label']} + "
              f"{_PATH_CONFIG[PATH2_ID]['label']}")

    from aioquic.quic.events import StreamDataReceived

    class PersistentClientProtocol(QuicConnectionProtocol):
        def quic_event_received(self, event):
            if isinstance(event, StreamDataReceived):
                try:
                    data = json.loads(event.data.decode("utf-8"))
                    print(f"\n[SERVER PUSH] Recommendation received: {data}")
                except Exception:
                    pass

    client_conf = QuicConfiguration(is_client=True, verify_mode=ssl.CERT_NONE)

    try:
        async with connect(_PATH_CONFIG[PATH1_ID]["host"], _PATH_CONFIG[PATH1_ID]["port"], configuration=client_conf, create_protocol=PersistentClientProtocol) as client1, \
                   connect(_PATH_CONFIG[PATH2_ID]["host"], _PATH_CONFIG[PATH2_ID]["port"], configuration=client_conf, create_protocol=PersistentClientProtocol) as client2:
            
            start_mono = time.monotonic()
            total_records = 0
            iteration = 0

            while True:
                elapsed = time.monotonic() - start_mono
                if elapsed >= duration_sec:
                    break

                cond1 = _sample_condition(cond1_base, elapsed, scenario)
                cond2 = _sample_condition(cond2_base, elapsed, scenario)

                rtt1 = _compute_rtt(cond1)
                rtt2 = _compute_rtt(cond2)

                # Send both paths concurrently — this is the dual-path MP-QUIC analogue.
                result1, result2 = await asyncio.gather(
                    asyncio.create_task(_send_on_path(PATH1_ID, rtt1, cond1, send_interval_sec, client1)),
                    asyncio.create_task(_send_on_path(PATH2_ID, rtt2, cond2, send_interval_sec, client2)),
                )

                total_records += 2
                iteration += 1

                if verbose:
                    p1_tag = f"RTT={result1['rtt_ms']:.1f}ms" if result1["status"] == "success" else result1["status"]
                    p2_tag = f"RTT={result2['rtt_ms']:.1f}ms" if result2["status"] == "success" else result2["status"]
                    degraded1 = " ⚠" if result1["rtt_ms"] > RTT_DEGRADATION_MS else ""
                    degraded2 = " ⚠" if result2["rtt_ms"] > RTT_DEGRADATION_MS else ""
                    print(
                        f"  [{elapsed:6.1f}s] "
                        f"P1({_PATH_CONFIG[PATH1_ID]['label']}): {p1_tag}{degraded1} | "
                        f"P2({_PATH_CONFIG[PATH2_ID]['label']}): {p2_tag}{degraded2} | "
                        f"records={total_records}",
                        end="\r",
                    )

                # Wait for the next send cycle.
                await asyncio.sleep(send_interval_sec)

        if verbose:
            print(f"\n  Done: {total_records} records logged for '{name}'")

        return total_records

    except Exception as e:
        print(f"\n[ERROR] Connection failed: {e}. Is mpquic_server.py running?")
        return 0


# ── Entry point ────────────────────────────────────────────────────────────────

async def _main(
    scenario_name: Optional[str],
    send_interval_sec: float,
) -> None:
    init_logger()

    scenarios_to_run = (
        [s for s in SCENARIOS if s["name"] == scenario_name]
        if scenario_name
        else SCENARIOS
    )

    if not scenarios_to_run:
        print(
            f"[ERROR] Scenario '{scenario_name}' not found.\n"
            f"Available: {[s['name'] for s in SCENARIOS]}"
        )
        sys.exit(1)

    total_duration = sum(s["duration_seconds"] for s in scenarios_to_run)
    print("=" * 60)
    print("  MP-QUIC Network Emulator (hardware-absent mode)")
    print("=" * 60)
    print(f"  Scenarios    : {len(scenarios_to_run)}")
    print(f"  Est. duration: ~{int(total_duration)}s ({int(total_duration / 60)} min)")
    print(f"  Send interval: {send_interval_sec}s per path")
    print(f"  Path 1       : {_PATH_CONFIG[PATH1_ID]['label']}  → QUIC UDP {SERVER_PORT_PATH1}")
    print(f"  Path 2       : {_PATH_CONFIG[PATH2_ID]['label']}  → QUIC UDP {SERVER_PORT_PATH2}")
    print(f"  RTT threshold: {RTT_DEGRADATION_MS} ms (⚠ = predicted degraded)")
    print("=" * 60)

    grand_total = 0
    for scenario in scenarios_to_run:
        grand_total += await run_scenario(scenario, send_interval_sec)

    print(f"\n[DONE] Total records logged: {grand_total}")
    print(f"       Log: data/logs/path_log2.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MP-QUIC network emulator — hardware-absent simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python simulator/network_emulator.py\n"
            "  python simulator/network_emulator.py --scenario normal\n"
            "  python simulator/network_emulator.py --interval 0.25\n"
        ),
    )
    parser.add_argument(
        "--scenario",
        metavar="NAME",
        default=None,
        help=(
            "Run a single named scenario instead of all 14. "
            f"Available: {[s['name'] for s in SCENARIOS]}"
        ),
    )
    parser.add_argument(
        "--interval",
        metavar="SEC",
        type=float,
        default=SEND_INTERVAL_SEC,
        help=f"Send interval in seconds per path (default: {SEND_INTERVAL_SEC}, from config.py/SEND_INTERVAL_SEC env)",
    )
    args = parser.parse_args()

    asyncio.run(_main(args.scenario, args.interval))


if __name__ == "__main__":
    main()
