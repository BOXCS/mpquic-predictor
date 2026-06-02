# Code Standards

## General

- Keep modules small and single-purpose; one file should own one responsibility
- Fix root causes — do not layer workarounds or silent fallbacks over broken logic
- Do not mix unrelated concerns in one module (e.g. sensor reading and path switching
  must not share the same file)
- All configuration values (IP addresses, ports, thresholds, intervals) must be sourced
  from `hardware/config.py` or `.env` — never hardcoded inline
- Use descriptive variable names that reflect domain meaning
  (e.g. `rtt_ms`, `goodput_mbps`, `path_label`) not generic names like `data` or `val`

## Python

- Target Python 3.10+ syntax throughout the project
- Use type hints on all function signatures; avoid untyped `dict` or `Any` where
  a dataclass or TypedDict can be used instead
- Validate all external input (sensor readings, MP-QUIC payloads, API request bodies)
  at system boundaries before passing them into business logic
- Use `async`/`await` consistently in all MP-QUIC and FastAPI components;
  do not mix blocking calls inside async functions
- Raise explicit, descriptive exceptions — do not silently swallow errors
  with bare `except` blocks
- Format all Python files with `black` and lint with `flake8`

## FastAPI

- Keep each route handler focused on a single responsibility;
  delegate logic to service modules (e.g. `predictor_service.py`, `metric_monitor.py`)
- Validate and parse all request bodies using Pydantic models before any logic runs
- Return consistent response shapes across all endpoints:
  `{ "status": "ok"|"error", "data": ..., "message": ... }`
- WebSocket handlers in `ws_broadcaster.py` must not contain business logic —
  they only push pre-computed data to connected clients
- Use dependency injection for database sessions; do not instantiate DB connections
  inside route functions directly

## React + Vite

- Prefer functional components with hooks; do not use class components
- Custom hooks (`useWebSocket.js`, `useMetrics.js`) own data-fetching logic —
  components only consume and render
- Do not fetch data directly inside JSX or `useEffect` without an abstraction layer;
  use the existing hooks or create a new one
- Keep page components (`Dashboard.jsx`, `History.jsx`, `Scenario.jsx`) layout-only;
  move reusable UI into `src/components/`
- WebSocket connection lifecycle must be managed inside `useWebSocket.js` only —
  no other file should open or close WebSocket connections

## Styling

- Use Tailwind CSS utility classes exclusively — no inline `style` attributes
  and no custom CSS files unless absolutely necessary
- Do not hardcode color hex values; use Tailwind semantic classes
  (e.g. `text-red-500`, `bg-green-100`) for status-driven coloring
- Chart components (Recharts) must receive data as props —
  no internal data fetching inside chart components

## API Routes

- All routes are defined under `web/backend/routes/`;
  do not define routes directly in `main.py`
- Input validation happens at the route layer via Pydantic before reaching any service
- Database reads and writes are performed only through SQLAlchemy ORM;
  do not write raw SQL strings in route handlers
- Return HTTP 422 for validation errors, 404 for missing resources,
  and 500 with a descriptive message for unexpected failures

## Data and Storage

- Network metrics (RTT, goodput, predictions, switching events) belong in `metrics.db`
- Sensor readings (temperature, humidity, timestamps) belong in `sensor_data.db`
- Simulation logs belong in `data/logs/path_log.csv` — append only, never overwrite
- Model artifacts (weights, scaler, config) belong in `model/saved/rtt only` —
  written only by `model/train.py`, never by runtime components
- Do not store raw sensor payloads or large numpy arrays directly in SQLite;
  aggregate or summarize before writing

## File Organization

- `hardware/`   — Raspberry Pi scripts only: sensor reading, MP-QUIC client,
                  path switcher, and hardware config
- `simulator/`  — Data generation and network emulation scripts only;
                  no model or server logic here
- `model/`      — All ML code: training, inference, TFLite export, and saved artifacts;
                  no HTTP or database logic here
- `server/`     — Server-side runtime: MP-QUIC receiver, metric monitor,
                  predictor service, and DB writer; no web framework code here
- `web/backend/` — FastAPI app, routes, WebSocket broadcaster;
                   no direct MP-QUIC or model inference logic here
- `web/frontend/src/pages/` — Page-level layout components only
- `web/frontend/src/components/` — Reusable UI components (charts, banners, status cards)
- `web/frontend/src/hooks/` — All data-fetching and WebSocket logic
- `evaluation/` — Standalone evaluation and visualization scripts only;
                  must not import from `server/` or `web/`
- `data/`       — Storage only; no executable logic in this directory
