"""
server/predictor_service.py — LSTM Inference Engine

Runs asynchronously on a fixed timer. Pulls the sliding window from metric_monitor,
runs LSTM inference, predicts degradation, logs to database, and pushes
switching recommendations to connected clients via mpquic_server if needed.

Guards against bad switching decisions:
  1. Warm-up guard      — skip inference until the new active path's window
                          is fully repopulated with fresh post-switch data.
  2. Hysteresis guard   — only switch if the alternative path's average RTT
                          over the last hysteresis window is at least
                          SWITCH_MARGIN_PCT % lower than the active path's.
  3. Cooldown guard     — enforce a minimum SWITCH_COOLDOWN_SEC between
                          consecutive switches.
  4. Initial path       — at startup, compare both paths' average RTT and
                          activate the lower-RTT path.
"""

import os
import json
import time
import pickle
import asyncio
import logging
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

from hardware.config import INFERENCE_INTERVAL_SEC, PATH1_ID, PATH2_ID, RTT_DEGRADATION_MS
from server.metric_monitor import monitor
from server.db_writer import write_prediction, write_switching_event
from server.mpquic_server import active_connections

logger = logging.getLogger("PredictorService")
logging.basicConfig(level=logging.INFO)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model', 'saved', 'rtt only')

# ── Env-configurable hysteresis parameters ─────────────────────────────────────
# SWITCH_MARGIN_PCT: the alternative path must have an average RTT that is this
#   percent lower than the active path before a switch is triggered.
#   Default: 20  (i.e. alt_rtt must be ≤ 0.80 × active_rtt)
SWITCH_MARGIN_PCT: float = float(os.getenv("SWITCH_MARGIN_PCT", "20"))

# SWITCH_COOLDOWN_SEC: minimum seconds between two consecutive switches.
#   Default: 30
SWITCH_COOLDOWN_SEC: float = float(os.getenv("SWITCH_COOLDOWN_SEC", "30"))


def _avg_rtt(history: list, n: int) -> float | None:
    """Return average RTT over the last *n* samples, or None if insufficient data."""
    samples = [r["rtt_ms"] for r in history if r.get("rtt_ms", 0) > 0]
    if len(samples) < n:
        return None
    return float(np.mean(samples[-n:]))


class PredictorService:
    def __init__(self):
        logger.info("Loading LSTM model and scaler...")
        self.model = load_model(os.path.join(MODEL_DIR, 'lstm_model.keras'))
        with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'config.pkl'), 'rb') as f:
            self.config = pickle.load(f)

        self.window_size = self.config['window_size']
        self.features    = self.config['features']
        self.pred_threshold = self.config['pred_threshold']

        # Hysteresis window: half the model window (sustained improvement check)
        self.hysteresis_n = max(1, self.window_size // 2)

        # ── State ─────────────────────────────────────────────────────────────
        # active_path is initially undetermined; resolved on first full window.
        self.active_path: int | None = None
        self.last_switch_time: float = 0.0          # monotonic timestamp
        self.samples_since_switch: int = self.window_size  # start warm

        logger.info(
            f"Predictor initialized. Window: {self.window_size}, "
            f"Threshold: {self.pred_threshold:.4f}, "
            f"Cooldown: {SWITCH_COOLDOWN_SEC:.1f}s, "
            f"Margin: {SWITCH_MARGIN_PCT:.1f}%, "
            f"Hysteresis N: {self.hysteresis_n}"
        )

    # ── Feature extraction ─────────────────────────────────────────────────────

    def _extract_features(self, history: list) -> pd.DataFrame | None:
        if len(history) < self.window_size:
            return None

        df = pd.DataFrame(history)

        # Map fields
        if 'packet_loss_pct' not in df.columns and 'loss_pct' in df.columns:
            df['packet_loss_pct'] = df['loss_pct']

        if 'status_enc' not in df.columns:
            df['status_enc'] = df['rtt_ms'].apply(
                lambda rtt: 1 if 0 < rtt < RTT_DEGRADATION_MS else 0
            )

        df['rtt_roll_mean'] = df['rtt_ms'].rolling(5, min_periods=1).mean()
        df['rtt_roll_std']  = df['rtt_ms'].rolling(5, min_periods=1).std().fillna(0)
        df['rtt_diff']      = df['rtt_ms'].diff().fillna(0)

        for f in self.features:
            if f not in df.columns:
                df[f] = 0.0

        return df[self.features].tail(self.window_size)

    # ── QUIC broadcast ─────────────────────────────────────────────────────────

    async def _broadcast_switch(self, target_path: int, reason: str):
        payload = {"action": "switch", "to_path": target_path, "reason": reason}
        payload_bytes = (json.dumps(payload) + "\n").encode("utf-8")

        for protocol in list(active_connections):
            try:
                stream_id = protocol._quic.get_next_available_stream_id()
                protocol._quic.send_stream_data(stream_id, payload_bytes, end_stream=True)
                protocol.transmit()
            except Exception as e:
                logger.error(f"Failed to push recommendation to {protocol._client_ip}: {e}")

    # ── Hysteresis check ───────────────────────────────────────────────────────

    def _alt_path_is_better(self, metrics: dict) -> tuple[bool, int, str]:
        """
        Return (should_switch, target_path, reason).

        Conditions for a switch:
          - Both paths must have at least hysteresis_n valid RTT samples.
          - The alternative path's average RTT must be at least
            SWITCH_MARGIN_PCT % lower than the active path's average RTT.
        """
        alt_path = PATH2_ID if self.active_path == PATH1_ID else PATH1_ID

        active_rtt = _avg_rtt(metrics[self.active_path], self.hysteresis_n)
        alt_rtt    = _avg_rtt(metrics[alt_path],          self.hysteresis_n)

        if active_rtt is None or alt_rtt is None:
            return False, alt_path, "insufficient data"

        # Require alt to be at least SWITCH_MARGIN_PCT % lower
        threshold_rtt = active_rtt * (1.0 - SWITCH_MARGIN_PCT / 100.0)
        if alt_rtt <= threshold_rtt:
            reason = (
                f"Hysteresis: alt_path{alt_path} avg_rtt={alt_rtt:.1f}ms "
                f"is {((active_rtt - alt_rtt) / active_rtt * 100):.1f}% "
                f"lower than active_path{self.active_path} avg_rtt={active_rtt:.1f}ms "
                f"(required >{SWITCH_MARGIN_PCT:.0f}%)"
            )
            return True, alt_path, reason

        return False, alt_path, (
            f"Hysteresis not met: alt={alt_rtt:.1f}ms vs active={active_rtt:.1f}ms "
            f"(need {SWITCH_MARGIN_PCT:.0f}% improvement)"
        )

    # ── Main loop ──────────────────────────────────────────────────────────────

    async def run_loop(self):
        logger.info(f"Starting PredictorService loop (Interval: {INFERENCE_INTERVAL_SEC}s)")
        while True:
            await asyncio.sleep(INFERENCE_INTERVAL_SEC)

            metrics = monitor.get_latest_metrics()

            # ── Bug 1 fix: initial active path selection ──────────────────────
            # On startup (active_path is None), wait until both windows have
            # enough data, then activate the path with lower average RTT.
            if self.active_path is None:
                rtt1 = _avg_rtt(metrics[PATH1_ID], self.window_size)
                rtt2 = _avg_rtt(metrics[PATH2_ID], self.window_size)
                if rtt1 is None or rtt2 is None:
                    logger.debug("Waiting for initial data on both paths…")
                    continue
                self.active_path = PATH1_ID if rtt1 <= rtt2 else PATH2_ID
                self.samples_since_switch = self.window_size  # already warm
                logger.info(
                    f"Initial path selected: path {self.active_path} "
                    f"(P1 avg_rtt={rtt1:.1f}ms, P2 avg_rtt={rtt2:.1f}ms)"
                )
                continue

            # ── Warm-up guard ─────────────────────────────────────────────────
            self.samples_since_switch += 1
            if self.samples_since_switch < self.window_size:
                logger.debug(
                    f"Warming up path {self.active_path}: "
                    f"{self.samples_since_switch}/{self.window_size} fresh samples"
                )
                continue

            # ── LSTM inference on active path ─────────────────────────────────
            active_history = metrics[self.active_path]
            features = self._extract_features(active_history)

            if features is None:
                continue  # Not enough data yet

            scaled = self.scaler.transform(features)
            X = scaled.reshape(1, self.window_size, len(self.features))
            prob = float(self.model.predict(X, verbose=0)[0][0])

            degraded = prob >= self.pred_threshold

            # Log prediction
            write_prediction(predicted_path=self.active_path, confidence=prob, degradation_detected=degraded)

            if not degraded:
                continue

            logger.warning(
                f"Path {self.active_path} degrading "
                f"(prob: {prob:.3f} >= {self.pred_threshold:.3f})"
            )

            # ── Cooldown guard ────────────────────────────────────────────────
            now = time.monotonic()
            elapsed = now - self.last_switch_time
            if elapsed < SWITCH_COOLDOWN_SEC:
                remaining = SWITCH_COOLDOWN_SEC - elapsed
                logger.info(
                    f"Switch suppressed — cooldown active. "
                    f"{remaining:.1f}s remaining (cooldown={SWITCH_COOLDOWN_SEC:.1f}s)"
                )
                continue

            # ── Bug 2 fix: hysteresis guard ───────────────────────────────────
            # Only switch if the alternative path's average RTT is sustainably
            # SWITCH_MARGIN_PCT % lower (not just a single sample below threshold).
            should_switch, target_path, reason = self._alt_path_is_better(metrics)
            if not should_switch:
                logger.info(f"Switch suppressed — {reason}")
                continue

            # ── Perform switch ────────────────────────────────────────────────
            switch_reason = f"LSTM {prob:.3f}>={self.pred_threshold:.3f}; {reason}"
            await self._broadcast_switch(target_path, switch_reason)
            write_switching_event(
                from_path=self.active_path,
                to_path=target_path,
                reason=switch_reason,
            )

            logger.info(f"Switched: path {self.active_path} → path {target_path}")

            # Reset both cooldown counters
            self.active_path = target_path
            self.last_switch_time = time.monotonic()
            self.samples_since_switch = 0


predictor_service = PredictorService()


async def start_predictor_service():
    try:
        await predictor_service.run_loop()
    except Exception as e:
        logger.exception(f"Predictor service crashed: {e}")
