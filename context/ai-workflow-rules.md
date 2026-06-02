# AI Workflow Rules

## Approach

Build this project incrementally using a spec-driven workflow.
Context files define what to build, how to build it, and the
current state of progress. Always implement against these specs —
do not infer or invent behavior that is not explicitly defined.

The system has five distinct layers: hardware (Raspberry Pi),
transport (MP-QUIC), ML inference (LSTM/TFLite), server, and
web dashboard. Each layer has a clear boundary defined in
`architecture.md`. Work within one layer at a time unless
an integration step is explicitly scoped.

## Scoping Rules

- Work on one feature unit at a time
- Prefer small, verifiable increments over large speculative changes
- Do not combine unrelated system boundaries in a single
  implementation step
- Hardware-side changes (Raspberry Pi) and server-side changes
  must always be treated as separate implementation units
- Model artifacts in `model/saved/rtt only` are read-only at runtime;
  do not include retraining logic in any non-training component

## When to Split Work

Split an implementation step if it combines:

- Hardware-side logic (sensor reading, path switching) and
  server-side logic (metric monitoring, inference service)
- ML model changes and transport/protocol changes
- Multiple unrelated FastAPI routes or WebSocket handlers
- Frontend UI changes and backend data pipeline changes
- Any behavior not clearly defined in the context files

If a change cannot be verified end to end quickly,
the scope is too broad — split it.

## Handling Missing Requirements

- Do not invent product behavior not defined in the context files
- If a requirement is ambiguous, resolve it in the relevant
  context file before implementing
- If a requirement is missing, add it as an open question
  in `progress-tracker.md` before continuing
- Do not assume network conditions, path states, or sensor values —
  all thresholds and configurations must be sourced from
  `hardware/config.py` or `.env`

## Protected Files

Do not modify the following unless explicitly instructed:

- `model/saved/rtt only` — trained model artifacts (lstm_model.keras,
  scaler.pkl, config.pkl); only updated via `model/train.py`
- `data/logs/path_log.csv` — existing simulation log;
  append-only, never overwrite historical entries
- `simulator/data_generator.py` — finalized 14-scenario
  degradation generator; treat as stable reference data
- `evaluation/compare.py` and `evaluation/test_prediction.py` —
  finalized evaluation scripts; do not alter baselines

## Keeping Docs in Sync

Update the relevant context file whenever implementation changes:

- System architecture or layer boundaries → `architecture.md`
- Storage schema or new database tables → `architecture.md`
- New degradation scenarios or model parameters → `data-models.md`
- Feature progress or newly completed units → `progress-tracker.md`
- Code conventions or new patterns introduced → `conventions.md`

## Before Moving to the Next Unit

1. The current unit works end to end within its defined scope
2. No invariant defined in `architecture.md` was violated
3. `progress-tracker.md` reflects the completed work
4. Python components pass: `pytest` runs without errors
5. If frontend was modified: `npm run build` inside `web/frontend/`
   completes without errors
6. No hardcoded IP addresses, ports, or thresholds —
   all configuration values are sourced from `config.py` or `.env`
