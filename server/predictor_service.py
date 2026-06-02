"""
server/predictor_service.py — LSTM Inference Engine

Runs asynchronously on a fixed timer. Pulls the sliding window from metric_monitor,
runs LSTM inference, predicts degradation, logs to database, and pushes
switching recommendations to connected clients via mpquic_server if needed.
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

class PredictorService:
    def __init__(self):
        logger.info("Loading LSTM model and scaler...")
        self.model = load_model(os.path.join(MODEL_DIR, 'lstm_model.keras'))
        with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'config.pkl'), 'rb') as f:
            self.config = pickle.load(f)

        self.window_size = self.config['window_size']
        self.features = self.config['features']
        self.pred_threshold = self.config['pred_threshold']
        
        self.active_path = PATH1_ID
        logger.info(f"Predictor initialized. Window: {self.window_size}, Threshold: {self.pred_threshold:.4f}")

    def _extract_features(self, history: list) -> np.ndarray:
        if len(history) < self.window_size:
            return None
        
        df = pd.DataFrame(history)
        
        # Map fields
        if 'packet_loss_pct' not in df.columns and 'loss_pct' in df.columns:
            df['packet_loss_pct'] = df['loss_pct']
            
        if 'status_enc' not in df.columns:
            # Assume 1 for success (RTT < degradation), 0 otherwise or timeout
            df['status_enc'] = df['rtt_ms'].apply(lambda rtt: 1 if 0 < rtt < RTT_DEGRADATION_MS else 0)

        df['rtt_roll_mean'] = df['rtt_ms'].rolling(5, min_periods=1).mean()
        df['rtt_roll_std']  = df['rtt_ms'].rolling(5, min_periods=1).std().fillna(0)
        df['rtt_diff']      = df['rtt_ms'].diff().fillna(0)
        
        # Ensure all required features are present
        for f in self.features:
            if f not in df.columns:
                df[f] = 0.0
                
        window_df = df[self.features].tail(self.window_size)
        return window_df.values

    async def _broadcast_switch(self, target_path: int, reason: str):
        payload = {
            "action": "switch",
            "to_path": target_path,
            "reason": reason
        }
        payload_bytes = (json.dumps(payload) + "\n").encode("utf-8")
        
        for protocol in list(active_connections):
            try:
                stream_id = protocol._quic.get_next_available_stream_id()
                protocol._quic.send_stream_data(stream_id, payload_bytes, end_stream=True)
                protocol.transmit()
            except Exception as e:
                logger.error(f"Failed to push recommendation to {protocol._client_ip}: {e}")

    async def run_loop(self):
        logger.info(f"Starting PredictorService loop (Interval: {INFERENCE_INTERVAL_SEC}s)")
        while True:
            await asyncio.sleep(INFERENCE_INTERVAL_SEC)
            
            metrics = monitor.get_latest_metrics()
            
            # We predict specifically on the currently active path
            active_history = metrics[self.active_path]
            features = self._extract_features(active_history)
            
            if features is None:
                continue # Not enough data yet
                
            scaled = self.scaler.transform(features)
            X = scaled.reshape(1, self.window_size, len(self.features))
            prob = float(self.model.predict(X, verbose=0)[0][0])
            
            degraded = prob >= self.pred_threshold
            
            # Log prediction
            write_prediction(predicted_path=self.active_path, confidence=prob, degradation_detected=degraded)
            
            if degraded:
                logger.warning(f"Path {self.active_path} degrading (prob: {prob:.3f} >= {self.pred_threshold:.3f})")
                
                # Switch to alternate path
                target_path = PATH2_ID if self.active_path == PATH1_ID else PATH1_ID
                reason = f"LSTM Prediction: {prob:.3f} >= threshold"
                
                await self._broadcast_switch(target_path, reason)
                write_switching_event(from_path=self.active_path, to_path=target_path, reason=reason)
                
                logger.info(f"Switched active path from {self.active_path} to {target_path}")
                self.active_path = target_path

predictor_service = PredictorService()

async def start_predictor_service():
    try:
        await predictor_service.run_loop()
    except Exception as e:
        logger.exception(f"Predictor service crashed: {e}")
