# Progress Tracker

Update this file after every meaningful implementation change.

---

## Current Phase

**Phase 6 — Hardware Integration (Raspberry Pi)**

---

## Current Goal

Phase 6 — Hardware Integration is **in progress**.
`hardware/mpquic_client.py` is complete and ready for on-Pi testing with the server.
Next step: implement `hardware/path_switcher.py` (socket-bind mechanism).

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

### Hardware Config (Phase 6) ✅
- `hardware/config.py` — fully updated for real hardware with all Phase 6
  resolved values:
  - `SERVER_HOST=192.168.1.10` (LAN IP of dev machine)
  - `ALLOWED_CLIENT_IPS=['192.168.1.18']` (Pi's LAN IP, parsed as list)
  - `SERVER_PORT_PATH1=5001`, `SERVER_PORT_PATH2=5002` (UDP, firewall open)
  - `DHT_GPIO_PIN=4`, `DHT_SENSOR_TYPE=DHT11` (3-pin module, built-in pull-up)
  - `PATH_SWITCH_MECHANISM=socket_bind` (application-level; no sudo)
  - `SWITCH_MARGIN_PCT=20`, `SWITCH_COOLDOWN_SEC=30` (hysteresis)
  - Added `path1_addr()` / `path2_addr()` returning `(host, port)` tuples
    for use by the QUIC client
- `.env` (server) — updated with all Phase 6 hardware values; now fully
  defines `SERVER_HOST`, `ALLOWED_CLIENT_IPS`, `INFERENCE_INTERVAL_SEC`,
  and all switching hysteresis parameters

### Hardware — Sensor Reader (Phase 6) ✅
- `hardware/sensor_reader.py` — DHT11 sensor reader, Pi-only:
  - Single public function: `read_sensor() -> tuple[float | None, float | None]`
  - Up to 3 retries with 0.5 s delay between each on `RuntimeError`
  - Returns `(None, None)` on persistent failure — never raises or crashes
  - `_PIN_MAP` resolves BCM int to `board.D<n>` pin object
  - `use_pulseio=False` avoids root requirement on Pi OS
  - Standalone loop (`python3 -m hardware.sensor_reader`) prints readings
    at `SENSOR_INTERVAL_SEC` cadence for on-Pi wiring verification
  - All config sourced from `hardware/config.py` — no hardcoded values

### Hardware — QUIC Client (Phase 6) ✅
- `hardware/mpquic_client.py` — dual-path MP-QUIC telemetry client, Pi-only:
  - Persistent QUIC connection per active path via `aioquic.asyncio.connect`
  - `MPQuicClientProtocol` handles outgoing streams and incoming server-push
    switching recommendations (`StreamDataReceived`)
  - Reads DHT11 sensor each tick; sends null-safe payload even on sensor failure
  - Payload schema: `{path_id, temperature, humidity, rtt_ms, loss_pct, timestamp_ms}`
    (matches `server/metric_monitor.py` expected fields)
  - Forwards server push recommendations to `path_switcher.handle_recommendation()`
    via lazy import (path_switcher not yet implemented — graceful fallback)
  - Path 2 (eth0/LAN) gated behind `--enable-path2` CLI flag or
    `ENABLE_PATH2=true` in `.env` — no code change needed when LAN cable connects
  - `asyncio.gather(return_exceptions=True)` ensures one path failure does not
    kill the other
  - All addresses sourced from `hardware/config.path1_addr()` / `path2_addr()`

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

- **Phase 6 — Hardware Integration**: `hardware/config.py`, `.env`,
  `hardware/sensor_reader.py`, and `hardware/mpquic_client.py` complete.
  Next: `hardware/path_switcher.py` (socket-bind switching mechanism).

---

## Next Up

### Phase 6 — Hardware Integration (Raspberry Pi)

1. ~~`hardware/sensor_reader.py`~~ ✅ done
2. ~~`hardware/mpquic_client.py`~~ ✅ done
3. **`hardware/path_switcher.py`** ← **implement next**
   - Receives switching recommendation dict from mpquic_client
   - Implements `handle_recommendation(rec: dict) -> None`
   - Uses `socket_bind` mechanism: records target interface for the
     client's next transmission (no OS routing change; no sudo)
   - Thread-safe: recommendation may arrive from the QUIC receive callback
     while the send loop is running
4. **End-to-end validation** — Pi → server → dashboard;
   verify RTTChart updates, PathStatus reflects real path, AlertBanner
   fires on first switch

---

## Open Questions

> [!NOTE]
> All pre-Phase-6 open questions are resolved. No blockers remaining.
> See Architecture Decisions for the resolved values.

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

### Environment variable status (`.env` as of 2026-06-12)

| Variable              | Defined in `.env`? | Current value              | Notes                                       |
|-----------------------|--------------------|----------------------------|---------------------------------------------|
| `CORS_ORIGINS`        | ✅ Yes             | `http://localhost:5173,...` | Web backend CORS allowlist                 |
| `SERVER_HOST`         | ✅ Yes             | `192.168.1.10`             | LAN IP of dev machine (server)              |
| `SERVER_PORT_PATH1`   | ✅ Yes             | `5001`                     | UDP; firewall confirmed open                |
| `SERVER_PORT_PATH2`   | ✅ Yes             | `5002`                     | UDP; firewall confirmed open                |
| `ALLOWED_CLIENT_IPS`  | ✅ Yes             | `192.168.1.18`             | Raspberry Pi 4 LAN IP                       |
| `INFERENCE_INTERVAL_SEC` | ✅ Yes          | `1.0`                      | Explicit (was using default before)         |
| `SWITCH_MARGIN_PCT`   | ✅ Yes             | `20`                       | Hysteresis: alt must be ≥20 % lower RTT    |
| `SWITCH_COOLDOWN_SEC` | ✅ Yes             | `30`                       | Min seconds between switches                |

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
- **`mpquic_server.py` bind-address OSError (Phase 6)**: `aioquic.serve()`
  was called with `host=SERVER_HOST` (`192.168.1.10`). This is the
  client-facing LAN IP of the server machine — not a locally assignable
  address from the OS's perspective when the network interface binds.
  Fixed by changing both `serve()` calls to `host="0.0.0.0"` so the OS
  binds the UDP socket on all interfaces. `SERVER_HOST` is now used only
  by the Raspberry Pi client to know where to connect — never by the
  server-side bind call. Log messages updated to show both bind address
  and client-facing address clearly.
- **SQLite "database is locked" on Windows+WSL (Phase 6)**: when the server
  runs as Windows-native Python 3.11 but the project root is on the WSL
  filesystem (`\\wsl$\...`), SQLite file locking fails at the OS level
  because Windows and WSL use incompatible locking primitives on cross-OS
  mounts. Fixed in `server/db_writer.py`: the database directory is now
  read from the `DB_PATH` env var (default: project-relative `data/` on
  Linux). On Windows, `.env` sets `DB_PATH=C:/mpquic-data/` so both
  database files (`metrics.db`, `sensor_data.db`) live entirely on NTFS,
  which SQLite can lock correctly. The directory is created on startup via
  `os.makedirs(DATA_DIR, exist_ok=True)`. No ORM models or write functions
  were changed — only the path resolution block at the top of the module.


### Phase 6 resolved hardware values (2026-06-12)
- **Server LAN IP**: `192.168.1.10` → `SERVER_HOST`
- **Pi LAN IP**: `192.168.1.18` → `ALLOWED_CLIENT_IPS`
- **UDP ports**: 5001 (wlan0), 5002 (eth0) → open in firewall ✅
- **Sensor**: DHT11, BCM pin 4, 3-pin module with built-in pull-up
- **Path switching**: application-level socket binding (`socket_bind`); no sudo
- **Inference**: server-side only; TFLite on Pi is fallback only
- **Pi model**: Raspberry Pi 4 (not Pi 5 — note: `architecture.md` still says
  "Raspberry Pi 5"; update that doc if it matters for the thesis)

### Next step
Implement `hardware/path_switcher.py` (socket-bind mechanism, no sudo).
This file runs **on the Raspberry Pi only** — do not import it server-side.
