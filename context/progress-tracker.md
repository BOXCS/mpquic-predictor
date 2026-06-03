# Progress Tracker

Update this file after every meaningful implementation change.

---

## Current Phase

**Phase 6 — Hardware Integration (Raspberry Pi)**

---

## Current Goal

Integrate the Raspberry Pi hardware layer with the working server stack.
The full simulator pipeline (Phases 1–5) is verified stable.
Next session begins Phase 6: deploy the server on the dev machine,
connect the Pi as a dual-path QUIC client, and validate end-to-end
telemetry + switching on real hardware.

---

## Completed

### Project Setup
- `requirements.txt` — all dependencies pinned to specific versions, grouped
  by layer (simulator, model, transport, server, web, hardware/Raspberry Pi);
  includes `aioquic==1.3.0`, `fastapi==0.136.3`, `uvicorn==0.34.3`,
  `SQLAlchemy==2.0.50`, `pydantic==2.13.4`, `websockets==14.2`,
  `python-dotenv==1.2.2`; Raspberry Pi-only packages commented out
- `SETUP.md` — system dependencies, installation instructions, .env setup,
  and verification steps documented across Linux/macOS/Windows
- `.gitignore` — comprehensive ignores across Python, Node, Vite, secrets,
  and model artifacts created and verified

### Simulator (Phase 1–2) ✅ — 5/12 scenarios verified stable
- `simulator/data_generator.py` — 14 degradation scenarios fully generated
- `simulator/network_emulator.py` — async dual-path emulator (hardware-absent
  substitute for Raspberry Pi); replays all 14 scenarios via asyncio.gather;
  logs via logger/path_logger.py; all config sourced from hardware/config.py
- Scenarios verified stable (no ping-pong after hysteresis fix):
  1. `normal` — baseline, both paths healthy
  2. `path1_degrading` — gradual RTT increase on wlan0, eth0 stable
  3. `path1_degraded` — wlan0 fully degraded, eth0 carries traffic
  4. `path2_degrading` — partial coverage verified
  5. `both_degraded` — verified suppressed switching (hysteresis holds)

### Model (Phase 3)
- `model/train.py` — LSTM training pipeline, finalized
- `model/predictor.py` — real-time inference, finalized and fixed
- `model/saved/rtt only` — lstm_model.keras, scaler.pkl, config.pkl produced
- `model/export_tflite.py` — TFLite conversion finalized;
  output: `model/saved/lstm_model.tflite` (141.1 KB, SELECT_TF_OPS + Flex delegate)

### Hardware Config
- `hardware/config.py` — all IP addresses, ports, intervals, and thresholds
  defined; all values sourced from env vars with sensible defaults

### Server Layer
- `server/mpquic_server.py` — QUIC UDP server implemented using `aioquic`,
  listening on dual paths, parsing JSON streams, and featuring auto-generated
  TLS certificates and IP allowlisting.
- `server/metric_monitor.py` — computes RTT and goodput from payloads,
  maintains sliding windows, dynamically loads window size from `config.pkl`.
- `server/db_writer.py` — SQLAlchemy ORM implementation that cleanly separates
  and writes network metrics and sensor data to their respective SQLite databases.
- `server/predictor_service.py` — LSTM inference loop with full switching
  governance (warm-up guard, hysteresis guard, cooldown guard, RTT-based
  initial path selection). See Architecture Decisions for full detail.

### Web Backend
- `web/backend/main.py` — FastAPI app initialised with `lifespan` startup hook;
  verifies `metrics.db` + `sensor_data.db` connectivity on boot; registers all
  three route routers (`/metrics`, `/predictions`, `/scenarios`) and the
  WebSocket broadcaster (`/ws`); CORS middleware configured from `.env`;
  no business logic in this file.
- `web/backend/deps.py` — SQLAlchemy session dependency injectors for both DBs.
- `web/backend/ws_broadcaster.py` — WebSocket client registry and background
  `start_push_loop()`; broadcasts JSON payloads to connected clients every 1 second.
- `web/backend/routes/metrics.py` — `GET /metrics/`, `GET /metrics/summary`,
  `GET /metrics/switching-events`. Pydantic models on all responses.
- `web/backend/routes/predictions.py` — `GET /predictions/latest`,
  `GET /predictions/history`, `GET /predictions/active-path`.
- `web/backend/routes/scenarios.py` — `GET /scenarios/`, `GET /scenarios/live`,
  `GET /scenarios/{name}`.
- All 14 HTTP test cases verified returning `status='ok'` with correct key structure.

### Web Frontend (Phase 4) ✅
- Vite + React project, Tailwind CSS, React Router v7, Recharts, Lucide
- `src/hooks/useWebSocket.js` — WS lifecycle, exponential backoff reconnect,
  dynamic URL from `window.location.host` (proxied through Vite)
- `src/hooks/useMetrics.js` — 60-point FIFO per-path rolling window, deduplication
- `src/components/RTTChart.jsx` — dual-path live RTT chart with degradation tint
- `src/components/PathStatus.jsx` — per-path health cards with Lucide icons
- `src/components/AlertBanner.jsx` — dismissible slide-in switching event banner
- `src/components/Navbar.jsx` — live connection dot, active page highlighting
- `src/pages/Dashboard.jsx` — stat cards + chart + status grid
- `src/pages/History.jsx` — tabbed switching events / metrics history table
- `src/pages/Scenario.jsx` — scenario grid with live scenario highlight
- `vite.config.js` — proxy `/ws`, `/metrics`, `/predictions`, `/scenarios`
  to `127.0.0.1:8000` so the browser always routes through the Vite dev server
- **`npm run build`: 2316 modules, zero errors.**

### Evaluation (Phase 5) ✅
- `evaluation/compare.py` — LSTM vs Round Robin comparison, finalized
- `evaluation/test_prediction.py` — early warning test, finalized
- `evaluation/visualize_results.py` — generates 4 publication-ready charts:
  `rtt_comparison.png`, `degradation_prediction.png`,
  `switching_events.png`, `goodput_comparison.png`
- `logger/path_logger.py` — CSV log writer, finalized
- `data/logs/path_log.csv` — simulation log populated

---

## In Progress

- Phase 6 — Hardware Integration (starts next session)

---

## Next Up

### Phase 6 — Hardware Integration (Raspberry Pi)

1. **`hardware/sensor_reader.py`** — read DHT22 temperature/humidity sensor
   and forward readings to the MP-QUIC client payload
2. **`hardware/mpquic_client.py`** — dual-path QUIC client running on the Pi;
   sends telemetry concurrently over wlan0 (UDP 5001) and eth0 (UDP 5002)
3. **`hardware/path_switcher.py`** — receives switching recommendation from
   server and executes OS-level path switch (ip rule / route manipulation)
4. **End-to-end validation** — Pi connected to dev-machine server;
   verify telemetry arrives, inference runs, switches are executed correctly

---

## Open Questions

> [!IMPORTANT]
> These must be answered before Phase 6 begins.

### Network topology
- **What is the server's LAN IP?** The Pi needs `SERVER_HOST` set to the actual
  LAN IP of the dev machine (not `127.0.0.1`). This must be confirmed and added
  to `.env` as `SERVER_HOST=<LAN_IP>` before the Pi can connect.
- **Are ports 5001 and 5002 (UDP) open on the dev machine's firewall?**
  `aioquic` listens on UDP — `ufw allow 5001/udp && ufw allow 5002/udp` may be needed.
- **Does the Pi have a static IP or DHCP?** `mpquic_server.py` uses an IP
  allowlist (`ALLOWED_CLIENT_IPS`). The Pi's IP must be added to `.env` as
  `ALLOWED_CLIENT_IPS=<pi_ip>` before connections will be accepted.

### Hardware
- **Which GPIO pin is the DHT22 data line connected to?**
  `hardware/sensor_reader.py` needs the exact BCM pin number.
- **Is the DHT22 a DHT22 or AM2302?** The `adafruit_dht` driver init call
  differs between the two variants.
- **Which wlan0 SSID will the Pi connect to during the test?**
  Needed to confirm path 1 is consistently reachable.

### Path switching mechanism
- **What OS-level command should `path_switcher.py` run?**
  Options: `ip rule add`, `ip route`, `iptables MARK`, or application-level
  socket binding. Confirm which is appropriate for the Pi's OS version.
- **Does the Pi need `sudo` for routing commands, or will the process run as root?**

### Model deployment
- **Will inference run on the server (current default) or on the Pi via TFLite?**
  Currently `server/predictor_service.py` runs inference server-side. If the Pi
  needs local inference, `model/saved/lstm_model.tflite` + Flex delegate must be
  installed on the Pi (`tensorflow` or `tflite-runtime` with custom ops support).

---

## Architecture Decisions

- **SQLite over PostgreSQL**: chosen for simplicity and portability
  in a single-machine lab setup; no multi-user concurrency required
- **Two separate databases** (`metrics.db`, `sensor_data.db`): kept separate
  to isolate network telemetry from raw sensor data, making each easier
  to query independently during evaluation
- **Server-side inference only**: LSTM runs on the server, not the Raspi,
  to avoid compute constraints on the Pi; TFLite is only a fallback
- **aioquic over third-party MP-QUIC lib**: chosen because it is the most
  mature Python QUIC implementation and supports multipath extensions
  via asyncio natively
- **FastAPI + WebSocket over polling**: real-time push avoids unnecessary
  HTTP polling load and keeps dashboard latency low
- **asyncio.Lock() removed from ws_broadcaster.py**: module-level Lock
  creation causes `RuntimeError` in Uvicorn `--reload` workers because the
  Lock is bound to a different event loop than the one that services requests.
  Python asyncio guarantees that `set.add()` / `set.discard()` are atomic
  within a single-threaded event loop, making the Lock redundant.
- **WebSocket endpoint registered directly on `app` (not via `APIRouter`)**:
  FastAPI's `APIRouter` does not forward WebSocket routes correctly in all
  Uvicorn configurations. Moving `@app.websocket("/ws")` into `create_app()`
  in `main.py` ensures reliable handshake acceptance.
- **Vite proxy for WebSocket + REST**: browser connects to the Vite dev server
  (`:5173`) which proxies `/ws`, `/metrics`, `/predictions`, `/scenarios` to
  the FastAPI backend (`:8000`). This avoids cross-origin WebSocket failures
  when running in a remote dev container where `:8000` is not reachable from
  the browser's `127.0.0.1`.
- **`predictor_service.py` switching governance** — three independent guards
  applied in order to prevent ping-pong switching:
  1. **Warm-up guard**: skip inference for `window_size` ticks after a switch
     so stale pre-switch samples are flushed from the sliding window.
  2. **Hysteresis guard** (`SWITCH_MARGIN_PCT`, default 20 %): the alternative
     path's average RTT over the last `window_size // 2` samples must be at
     least 20 % lower than the active path's before a switch is triggered.
     This prevents transient RTT spikes from causing a switch.
  3. **Cooldown guard** (`SWITCH_COOLDOWN_SEC`, default 30 s): a hard minimum
     time between consecutive switches, independent of RTT readings.
  4. **RTT-based initial path selection**: `active_path` starts as `None`.
     On the first tick where both paths have a full window, the path with the
     lower average RTT is activated. Prevents incorrect initial activation when
     wlan0 is already stressed at boot.

---

## Session Notes

### Stable constants (do not change without explicit instruction)
- Model training and evaluation scripts are finalized — treat them as stable
- `data/logs/path_log.csv` is append-only; never overwrite existing rows
- `model/saved/lstm_model.tflite` uses SELECT_TF_OPS (Flex delegate);
  on Raspberry Pi use full TF runtime or link `libtensorflowlite_flex`

### Environment variable status (`.env` as of 2026-06-04)

| Variable              | Defined in `.env`? | Current value         | Notes                                      |
|-----------------------|--------------------|-----------------------|--------------------------------------------|
| `CORS_ORIGINS`        | ✅ Yes             | `http://localhost:5173,http://127.0.0.1:5173` | Web backend CORS allowlist |
| `SWITCH_MARGIN_PCT`   | ✅ Yes             | `20`                  | Hysteresis: alt must be ≥20 % lower RTT   |
| `SWITCH_COOLDOWN_SEC` | ✅ Yes             | `30`                  | Min seconds between switches               |
| `INFERENCE_INTERVAL_SEC` | ❌ Not set      | `1.0` (default)       | Defined as default in `hardware/config.py` |
| `SERVER_HOST`         | ❌ Not set         | `127.0.0.1` (default) | **Must be set to LAN IP before Pi connects** |
| `SERVER_PORT_PATH1`   | ❌ Not set         | `5001` (default)      | UDP port for wlan0; open in firewall first |
| `SERVER_PORT_PATH2`   | ❌ Not set         | `5002` (default)      | UDP port for eth0; open in firewall first  |
| `ALLOWED_CLIENT_IPS`  | ❌ Not set         | `127.0.0.1` (default) | **Must include Pi's IP before connecting** |

> [!CAUTION]
> `SERVER_HOST` and `ALLOWED_CLIENT_IPS` **must** be set in `.env` before
> starting `python3 -m server.mpquic_server` for the hardware session.
> The server will silently reject all connections from the Pi otherwise.

### Bug fixes applied this session
- **WebSocket 1006**: module-level `asyncio.Lock()` in `ws_broadcaster.py`
  caused connection rejection before handshake under Uvicorn `--reload`.
  Fixed by removing the Lock and registering the endpoint on `app` directly.
- **WebSocket browser 127.0.0.1 failure**: browser tried to reach `:8000`
  directly. Fixed via Vite proxy + dynamic `window.location.host` URL.
- **Ping-pong switching (stale window)**: inference ran on pre-switch data.
  Fixed with `samples_since_switch` warm-up counter.
- **Ping-pong switching (no cooldown)**: no minimum time between switches.
  Fixed with `SWITCH_COOLDOWN_SEC` hard cooldown.
- **Wrong initial active path**: hardcoded `PATH1_ID` at startup.
  Fixed by comparing avg RTT of both paths on first full window.
- **No hysteresis**: single RTT spike triggered a switch.
  Fixed with `_alt_path_is_better()` requiring `SWITCH_MARGIN_PCT` % improvement.

### Next session start point
**Phase 6 — Hardware Integration with Raspberry Pi.**
Start by answering the Open Questions above (LAN IP, Pi IP, GPIO pin,
path-switcher mechanism), then set `.env` accordingly, then implement
`hardware/sensor_reader.py` → `hardware/mpquic_client.py` → `hardware/path_switcher.py`
in that order.
