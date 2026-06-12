# Architecture Context

## Stack

| Layer         | Technology                     | Role                                                       |
| ------------- | ------------------------------ | ---------------------------------------------------------- |
| IoT Hardware  | Raspberry Pi 4 + DHT11         | IoT sender node; reads temperature & humidity sensor data  |
| Transport     | aioquic + asyncio (MP-QUIC)    | Multipath QUIC over wlan0 + eth0 simultaneously            |
| ML Model      | TensorFlow/Keras + TFLite      | LSTM training & lightweight inference on edge (Raspi)      |
| Preprocessing | scikit-learn + pandas + NumPy  | Feature engineering, normalization, and model evaluation   |
| Backend API   | FastAPI + uvicorn              | REST API + WebSocket endpoints for the dashboard           |
| Database      | SQLite + SQLAlchemy            | Stores network metrics, predictions, and sensor logs       |
| Frontend      | React + Vite + Recharts        | Real-time dashboard for RTT and degradation visualization  |
| Styling       | Tailwind CSS                   | Dashboard UI styling                                       |
| Network Emu   | tc netem                       | Network degradation emulation in lab environment           |

## System Boundaries

- `hardware/`   — Reads DHT11 sensor data, sends it via MP-QUIC client (wlan0 + eth0),
                  and executes path switching based on server recommendations
- `simulator/`  — Generates data for 14 degradation scenarios and wraps tc netem
                  for network condition emulation during the lab phase
- `model/`      — Handles LSTM training, real-time inference, and TFLite export;
                  stores model artifacts (lstm_model.keras, scaler.pkl, config.pkl)
- `server/`     — Receives data from Raspi, computes metrics (RTT, goodput),
                  runs LSTM inference, sends switching recommendations, writes to DB
- `web/`        — FastAPI backend serves REST + WebSocket;
                  React frontend renders the dashboard and history in real time
- `data/`       — Stores simulation logs (CSV) and SQLite databases for metrics & sensor data
- `evaluation/` — Compares LSTM vs Round Robin performance, tests early warning accuracy,
                  and generates charts for the thesis report

## Storage Model

- **metrics.db (SQLite)**: Stores RTT, goodput, LSTM prediction results,
  and path switching events per data transmission session
- **sensor_data.db (SQLite)**: Stores temperature and humidity readings
  from the DHT11 sensor along with transmission timestamps
- **path_log.csv**: Tabular simulation log for degradation scenarios;
  used as evaluation input and source for thesis charts
- **saved/ (model artifacts)**: Stores lstm_model.keras, scaler.pkl,
  and config.pkl produced during training; updated only on retraining

## Auth and Access Model

- The system has no user authentication; dashboard access is assumed to run
  on a controlled local network (LAN)
- The Raspberry Pi connects to the server using the IP address configured
  in `hardware/config.py` and environment variables in `.env`
- The server only accepts MP-QUIC connections from known Raspi IP addresses;
  there are no token or session mechanisms between components
- Switching recommendations flow one-way from server to Raspi;
  the Raspi cannot modify server configuration

## Invariants

1. LSTM inference always runs server-side via `predictor_service.py`;
   the Raspberry Pi only runs TFLite as a local fallback when the server is unavailable
2. Every sensor data transmission must use both paths (wlan0 + eth0) simultaneously;
   single-path operation is not a normal system state
3. Path switching is only executed by `path_switcher.py` based on an explicit
   server recommendation; no automatic switching occurs without a model prediction
4. The LSTM model is never retrained at runtime;
   artifacts in `model/saved/rtt only` are read-only during system operation
5. SQLite databases are only written by `db_writer.py`;
   all other components access data exclusively through FastAPI routes
