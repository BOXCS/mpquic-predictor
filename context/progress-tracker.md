# Progress Tracker

Update this file after every meaningful implementation change.

---

## Current Phase

Phase 6 — Hardware Integration (pending)

---

## Current Goal

Phase 6 Hardware Integration is PENDING.
- Prepare deployment for Raspberry Pi.

---

## Completed

### Simulator
- `simulator/data_generator.py` — 14 degradation scenarios fully generated
- `simulator/network_emulator.py` — async dual-path emulator (hardware-absent
  substitute for Raspberry Pi); replays all 14 scenarios via asyncio.gather;
  logs via logger/path_logger.py; all config sourced from hardware/config.py

### Model
- `model/train.py` — LSTM training pipeline, finalized
- `model/predictor.py` — real-time inference, finalized and fixed
- `model/saved/rtt only` — lstm_model.keras, scaler.pkl, config.pkl produced
- `model/export_tflite.py` — TFLite conversion finalized;
  output: `model/saved/lstm_model.tflite` (141.1 KB, SELECT_TF_OPS + Flex delegate)

### Hardware Config
- `hardware/config.py` — all IP addresses, ports, intervals, and thresholds
  defined; all values sourced from env vars with sensible defaults

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

### Server Layer
- `server/mpquic_server.py` — QUIC UDP server implemented using `aioquic`,
  listening on dual paths, parsing JSON streams, and featuring auto-generated
  TLS certificates and IP allowlisting.
- `server/metric_monitor.py` — computes RTT and goodput from payloads,
  maintains sliding windows, dynamically loads window size from `config.pkl`.
- `server/db_writer.py` — SQLAlchemy ORM implementation that cleanly separates
  and writes network metrics and sensor data to their respective SQLite databases.
- [x] **`server/predictor_service.py`**:
  - Implemented background asyncio loop running at `INFERENCE_INTERVAL_SEC`.
  - Loaded `lstm_model.keras` and `scaler.pkl` accurately from `model/saved/rtt only`.
  - Fetched active path metrics from `metric_monitor`, executed inference.
  - Broadcasted switching recommendation via `mpquic_server.py` persistent connections.
  - Logged events seamlessly to `db_writer.py`.

### Web Backend
- `web/backend/main.py` — FastAPI app initialised with `lifespan` startup hook;
  verifies `metrics.db` + `sensor_data.db` connectivity on boot; registers all
  three route routers (`/metrics`, `/predictions`, `/scenarios`) and the
  WebSocket broadcaster (`/ws`); CORS middleware configured from `.env`;
  no business logic in this file.
- `web/backend/deps.py` — SQLAlchemy session dependency injectors for both DBs.
- `web/backend/ws_broadcaster.py` — thread-safe WebSocket client registry and
  background `start_push_loop()`; calls route handlers directly to fetch
  latest metrics, predictions, and switching events; broadcasts JSON payloads
  to connected clients every 1 second; handles client disconnects gracefully.
- `web/backend/routes/metrics.py` — Pydantic models on all responses.
  - `GET /metrics/` — latest RTT + goodput per path, filterable by `path_id`.
  - `GET /metrics/summary?window_minutes=N` — per-path avg/min/max RTT, avg goodput,
    avg loss, and sample count aggregated over the last N minutes (default 10).
  - `GET /metrics/switching-events` — switching event history, newest first.
- `web/backend/routes/predictions.py` — Pydantic models on all responses.
  - `GET /predictions/latest` — single most recent LSTM prediction.
  - `GET /predictions/history?offset=N&limit=N&degraded_only=bool` — paginated
    history returning `{ items, total, limit, offset }`.
  - `GET /predictions/active-path` — active path + per-path health state
    (latest RTT, goodput, loss, prediction confidence for each path).
- `web/backend/routes/scenarios.py` — Pydantic models with PathCondition sub-model.
  - `GET /scenarios/` — all 12 scenarios with `simulated` flag from CSV log.
  - `GET /scenarios/live` — nearest-catalogue-match estimate from live DB RTT.
  - `GET /scenarios/{name}` — single scenario detail + simulated flag (404 on unknown).
- All 14 HTTP test cases verified returning `status='ok'` with correct key structure.

### Evaluation
- `evaluation/compare.py` — LSTM vs Round Robin comparison, finalized
- `evaluation/test_prediction.py` — early warning test, finalized
- `logger/path_logger.py` — CSV log writer, finalized
- `data/logs/path_log.csv` — simulation log populated

### Phase 4 — Web Frontend (scaffold)
- `web/frontend/` — Vite + React project scaffolded via `create-vite` (React template).
- `tailwind.config.js` — content paths for `src/**/*.{js,jsx}`, dark mode `class` strategy,
  Inter + JetBrains Mono font families registered.
- `postcss.config.js` — tailwindcss + autoprefixer plugins configured.
- `src/index.css` — Tailwind directives + all 14 CSS custom-property design tokens from
  `ui-context.md` for both `:root` (light) and `.dark` (dark); `body` reset + `.metric-value` mono class.
- `index.html` — title `MP-QUIC Dashboard`, meta description, Google Fonts preconnect +
  Inter + JetBrains Mono stylesheet link.
- `src/App.jsx` — React Router v7 setup: `/` → Dashboard, `/history` → History,
  `/scenarios` → Scenario, catch-all redirect.
- **Packages installed**: `recharts ^3`, `lucide-react ^1`, `react-router-dom ^7`,
  `tailwindcss@3`, `postcss`, `autoprefixer`.
- **Placeholder files created** (scaffold only, no logic):
  - `src/hooks/useWebSocket.js` — WS lifecycle hook stub
  - `src/hooks/useMetrics.js` — data layer hook stub
  - `src/components/RTTChart.jsx` — Recharts chart stub
  - `src/components/PathStatus.jsx` — path badge stub
  - `src/components/AlertBanner.jsx` — dismissible alert stub
  - `src/pages/Dashboard.jsx` — real-time monitoring page stub
  - `src/pages/History.jsx` — history page stub
  - `src/pages/Scenario.jsx` — scenario list page stub
- `npm run dev` — **Vite v8.0.16 ready in 448 ms, zero errors**.
- `npm run build` — **27 modules, zero errors**, CSS 4.92 kB, JS 233 kB.

### Phase 4 — Hooks (useWebSocket + useMetrics)
- `web/frontend/.env` — `VITE_WS_URL=ws://127.0.0.1:8000/ws`,
  `VITE_API_URL=http://127.0.0.1:8000`.
- `src/hooks/useWebSocket.js` — Full lifecycle: auto-connect to `VITE_WS_URL`,
  exponential backoff reconnect (base 1 s, max 30 s, ±20 % jitter), unmount cleanup,
  filters out "connected" handshake — exposes `{ payload, isConnected, lastUpdated }`.
- `src/hooks/useMetrics.js` — Consumes `useWebSocket()`; maintains 60-point FIFO
  rolling window per path (`metrics`), derives `latestPerPath`, accumulates
  deduplicated `switchingEvents` (by event id), guards all payload access against null —
  exposes `{ metrics, latestPerPath, prediction, activePath, switchingEvents,
  isConnected, lastUpdated }`.
- `src/components/MetricsDebug.jsx` — Temporary debug component rendering raw
  `useMetrics()` output; wired into `Dashboard.jsx` for verification.
- **End-to-end test passed**: 3 consecutive WS update frames verified:
  `metrics=1 item, prediction=True, active_path=2, switching_events correct`.
- `npm run build` — **30 modules, zero errors**.

### Phase 4 — Reusable Components
- `src/components/RTTChart.jsx` — Recharts LineChart for both paths simultaneously.
  - `mergePathData()` aligns per-path arrays into a single time-indexed series.
  - Colours via `getComputedStyle` at render time: `--chart-path-0` (wlan0),
    `--chart-path-1` (eth0), `--chart-degraded` (reference line + background tint).
  - Background tints amber when `prediction.degradation_detected` is true.
  - Dashed `ReferenceLine` at `degradationThreshold` (default 150 ms).
  - Custom tooltip with CSS-var colours; `isAnimationActive={false}` for live data.
  - Empty state with `Activity` icon. Responsive width, fixed 260 px height.
- `src/components/PathStatus.jsx` — Per-path health card panel.
  - `deriveHealth()` maps `degradation_detected` + RTT threshold to:
    `healthy` / `degraded` / `critical` / `unknown`.
  - Icons: `Wifi` (wlan0), `Network` (eth0), `CheckCircle` / `AlertTriangle`
    / `XCircle` / `Activity` per health state.
  - Active path highlighted with `--accent-primary` border + ACTIVE pill.
  - Shows RTT, goodput (kbps), loss %, prediction confidence per path.
- `src/components/AlertBanner.jsx` — Dismissible event banner.
  - Detects new events via `shownIdRef` in a `useEffect` on `latestId`.
  - Shows from-path to-path, reason, confidence, timestamp.
  - `--state-warning` border + blended background; `ArrowLeftRight` icon.
  - Auto-dismisses via `setTimeout` (default 8 s); manual dismiss button.
  - Slide-in CSS keyframe animation.
- `src/pages/Dashboard.jsx` — Real-Time Monitoring Page layout.
  - Top row of Stat Cards displaying live RTT (wlan0/eth0), aggregated total Goodput,
    and Prediction Confidence (with warning highlight if degraded).
  - Two-column grid placing `RTTChart` alongside `AlertBanner` and `PathStatus`.
  - Fully bound to a single `useMetrics()` hook.
- `src/components/Navbar.jsx` — Shared top navigation bar.
  - Active page highlighting.
  - Live connection status dot (green=connected, red=disconnected) using `isConnected`.
- `src/pages/History.jsx` — Metric and Event History Page.
  - Tabbed interface switching between 'Switching Events' and 'Network Metrics'.
  - Full-width data tables displaying historical data fetched via REST API.
  - Empty state with `Clock` icon when no data exists.
- `src/pages/Scenario.jsx` — Degradation Scenario List Page.
  - Grid of ScenarioCards displaying path conditions and simulation status.
  - Highlights active live scenario with `--accent-primary` styling.
  - Auto-refreshes every 10 seconds via REST polling.
- `MetricsDebug.jsx` deleted as requested.
- **ESLint: zero errors.**
- **`npm run build`: 2316 modules transformed, zero errors.**

### Evaluation (Phase 5)
- `evaluation/visualize_results.py` — generates 4 publication-ready evaluation charts:
  - `rtt_comparison.png` (LSTM vs Round Robin baseline over time)
  - `degradation_prediction.png` (Actual RTT vs predicted probability)
  - `switching_events.png` (Timeline of path switches)
  - `goodput_comparison.png` (Grouped bar of avg goodput)

---

## In Progress

- None yet.

---

## Next Up

### Server Layer
1. `server/predictor_service.py` — run LSTM inference and send switching recommendation

### Hardware Layer
5. `hardware/sensor_reader.py` — read DHT22 and forward to MP-QUIC client
6. `hardware/mpquic_client.py` — send data via wlan0 + eth0 simultaneously
7. `hardware/path_switcher.py` — receive recommendation and execute path switching

### Web Layer
10. `web/backend/main.py` — FastAPI app with REST and WebSocket setup
11. `web/backend/routes/metrics.py` — expose RTT and goodput data
12. `web/backend/routes/predictions.py` — expose LSTM prediction results
13. `web/backend/routes/scenarios.py` — expose degradation scenario state
14. `web/backend/ws_broadcaster.py` — push real-time updates to frontend
15. `web/frontend/` — Dashboard, History, Scenario pages + RTTChart,
    PathStatus, AlertBanner components + useWebSocket, useMetrics hooks

### Evaluation
16. `evaluation/visualize_results.py` — generate final charts for thesis report

---

## Open Questions

- Should `predictor_service.py` run inference on every incoming packet
  or on a fixed time interval (e.g. every 1s)?
- What is the minimum RTT threshold that triggers a switching recommendation?
  Define in `hardware/config.py` or `.env`?
- Should the TFLite fallback on Raspberry Pi log its predictions separately,
  or merge them into the same `metrics.db` schema?
- Does the dashboard need authentication, or is LAN-only access sufficient
  for the thesis demo?
- What is the exact feature window size (number of timesteps) the LSTM
  expects as input during real-time inference?

---

## Architecture Decisions

- **SQLite over PostgreSQL**: chosen for simplicity and portability
  in a single-machine lab setup; no multi-user concurrency required
- **Two separate databases** (metrics.db, sensor_data.db): kept separate
  to isolate network telemetry from raw sensor data, making each easier
  to query independently during evaluation
- **Server-side inference only**: LSTM runs on the server, not the Raspi,
  to avoid compute constraints on the Pi; TFLite is only a fallback
- **aioquic over third-party MP-QUIC lib**: chosen because it is the most
  mature Python QUIC implementation and supports multipath extensions
  via asyncio natively
- **FastAPI + WebSocket over polling**: real-time push avoids unnecessary
  HTTP polling load and keeps dashboard latency low

---

## Session Notes

- Model training and evaluation scripts are finalized — treat them as stable;
  do not modify without explicit instruction
- `data/logs/path_log.csv` is append-only; never overwrite existing rows
- All thresholds and IP configs are not yet defined — must be settled
  in `hardware/config.py` before hardware layer implementation begins
- The next session should start with `server/mpquic_server.py` as the
  entry point for the server layer
- `model/saved/lstm_model.tflite` uses SELECT_TF_OPS (Flex delegate) because
  LSTM layers emit TensorListReserve ops with dynamic element shapes that cannot
  be lowered by TFLITE_BUILTINS alone. On Raspberry Pi, use the full TensorFlow
  Python runtime (`import tensorflow as tf; tf.lite.Interpreter`) or link
  `libtensorflowlite_flex` — do not attempt to run inference with a
  bare TFLite runtime that omits the Flex delegate.
