# model/predictor.py
import numpy as np
import pandas as pd
import pickle
import os
from tensorflow.keras.models import load_model
from datetime import datetime

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'saved')

class PathPredictor:
    def __init__(self):
        print("Loading LSTM model...")
        self.model  = load_model(os.path.join(MODEL_DIR, 'lstm_model.keras'))
        with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'config.pkl'), 'rb') as f:
            self.config = pickle.load(f)

        self.window       = self.config['window_size']
        self.features     = self.config['features']
        self.pred_thresh  = self.config['pred_threshold']
        self.rtt_thresh   = self.config['threshold_rtt']

        # Buffer per jalur — menyimpan history untuk input LSTM
        self.buffers = {1: [], 2: []}
        print(f"Model loaded. Window={self.window}, Threshold={self.pred_thresh:.3f}")

    def add_record(self, path_id, rtt_ms, throughput_bps, packet_loss_pct, status):
        """Tambahkan record baru ke buffer jalur"""
        record = {
            'rtt_ms':           min(rtt_ms, 500),  # cap outlier
            'throughput_bps':   throughput_bps,
            'packet_loss_pct':  packet_loss_pct,
            'status_enc':       1 if status == 'success' else 0
        }

        buf = self.buffers[path_id]
        buf.append(record)

        # Jaga ukuran buffer — simpan 2x window untuk rolling stats
        if len(buf) > self.window * 2:
            self.buffers[path_id] = buf[-(self.window * 2):]

    def _compute_features(self, path_id):
        """Hitung features dari buffer termasuk rolling stats"""
        buf = self.buffers[path_id]
        if len(buf) < self.window:
            return None

        df = pd.DataFrame(buf)
        df['rtt_roll_mean'] = df['rtt_ms'].rolling(5, min_periods=1).mean()
        df['rtt_roll_std']  = df['rtt_ms'].rolling(5, min_periods=1).std().fillna(0)
        df['rtt_diff']      = df['rtt_ms'].diff().fillna(0)

        # Ambil window terakhir
        window_df = df[self.features].tail(self.window)
        return window_df.values

    def predict(self, path_id):
        """Return prediksi untuk jalur tertentu"""
        features = self._compute_features(path_id)

        if features is None:
            return {
                'ready': False,
                'reason': f'Buffer belum cukup ({len(self.buffers[path_id])}/{self.window})'
            }

        # Scale dan predict
        scaled   = self.scaler.transform(features)
        X        = scaled.reshape(1, self.window, len(self.features))
        prob     = float(self.model.predict(X, verbose=0)[0][0])
        label    = 'degraded' if prob > self.pred_thresh else 'stable'

        # Quality score: inverse dari degradation probability
        quality  = round(1 - prob, 3)

        # Label granular
        if prob < 0.3:
            detail_label = 'stable'
        elif prob < 0.5:
            detail_label = 'degrading'
        elif prob <= self.pred_thresh:
            detail_label = 'degrading'
        else:
            detail_label = 'degraded'

        return {
            'ready':                  True,
            'degradation_probability': round(prob, 3),
            'quality_score':          quality,
            'label':                  detail_label,
            'confidence':             round(abs(prob - 0.5) * 2, 3)
        }

    def get_recommendation(self, pred1, pred2):
        if not pred1.get('ready') and not pred2.get('ready'):
            return {'preferred_path': 1, 'reason': 'warming_up', 'fallback_path': 2}
        if not pred1.get('ready'):
            return {'preferred_path': 2, 'reason': 'path1_warming_up', 'fallback_path': 2}
        if not pred2.get('ready'):
            return {'preferred_path': 1, 'reason': 'path2_warming_up', 'fallback_path': 1}
    
        q1 = pred1['quality_score']
        q2 = pred2['quality_score']
        c1 = pred1['confidence']
        c2 = pred2['confidence']
    
        CONFIDENCE_GATE = 0.5  # minimum confidence untuk trigger switch
    
        p1_degraded = pred1['label'] == 'degraded' and c1 >= CONFIDENCE_GATE
        p2_degraded = pred2['label'] == 'degraded' and c2 >= CONFIDENCE_GATE
    
        if p1_degraded and not p2_degraded:
            return {'preferred_path': 2, 'reason': 'path1_predicted_degraded', 'fallback_path': 1}
        elif p2_degraded and not p1_degraded:
            return {'preferred_path': 1, 'reason': 'path2_predicted_degraded', 'fallback_path': 2}
        elif p1_degraded and p2_degraded:
            preferred = 1 if q1 >= q2 else 2
            return {'preferred_path': preferred, 'reason': 'both_degraded_choose_better', 'fallback_path': preferred}
        else:
            preferred = 1 if q1 >= q2 else 2
            return {'preferred_path': preferred, 'reason': 'quality_based', 'fallback_path': 2 if preferred == 1 else 1}