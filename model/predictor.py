# model/predictor.py — FIXED VERSION
import numpy as np
import pandas as pd
import pickle
import os
from tensorflow.keras.models import load_model

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'saved/rtt only')

class PathPredictor:
    def __init__(self):
        print("Loading LSTM model...")
        self.model  = load_model(os.path.join(MODEL_DIR, 'lstm_model.keras'))
        with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'config.pkl'), 'rb') as f:
            self.config = pickle.load(f)

        self.window      = self.config['window_size']
        self.features    = self.config['features']
        self.pred_thresh = self.config['pred_threshold']
        self.rtt_thresh  = self.config['threshold_rtt']
        self.buffers     = {1: [], 2: []}

        print(f"Model loaded. Window={self.window}, Threshold={self.pred_thresh:.3f}")
        # FIX: tampilkan features yang aktif supaya mudah debug
        print(f"Features aktif: {self.features}")

    def add_record(self, path_id, rtt_ms, throughput_bps, status):
        """
        FIX: packet_loss_pct dihapus dari signature.
        Model baru tidak menggunakan packet_loss_pct sebagai fitur
        karena berkorelasi sempurna dengan label dan menjadi shortcut.
        throughput_bps sekarang wajib dinamis — jangan hardcode 312.
        """
        record = {
            'rtt_ms':         min(rtt_ms, 500),
            'throughput_bps': throughput_bps,
            'status_enc':     1 if status == 'success' else 0
        }
        buf = self.buffers[path_id]
        buf.append(record)
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

        # Pastikan kolom yang diambil sesuai dengan self.features dari config
        # Ini otomatis konsisten dengan apapun yang dipakai saat training
        window_df = df[self.features].tail(self.window)
        return window_df.values

    def predict(self, path_id):
        """Return prediksi degradasi untuk jalur tertentu"""
        features = self._compute_features(path_id)

        if features is None:
            return {
                'ready':  False,
                'reason': f'Buffer belum cukup ({len(self.buffers[path_id])}/{self.window})'
            }

        scaled = self.scaler.transform(features)
        X      = scaled.reshape(1, self.window, len(self.features))
        prob   = float(self.model.predict(X, verbose=0)[0][0])

        if prob < 0.3:
            detail_label = 'stable'
        elif prob < self.pred_thresh:
            detail_label = 'degrading'
        else:
            detail_label = 'degraded'

        return {
            'ready':                   True,
            'degradation_probability': round(prob, 3),
            'quality_score':           round(1 - prob, 3),
            'label':                   detail_label,
            'confidence':              round(abs(prob - 0.5) * 2, 3)
        }

    def get_recommendation(self, pred1, pred2):
        if not pred1.get('ready') and not pred2.get('ready'):
            return {'preferred_path': 1, 'reason': 'warming_up', 'fallback_path': 2}
        if not pred1.get('ready'):
            return {'preferred_path': 2, 'reason': 'path1_warming_up', 'fallback_path': 2}
        if not pred2.get('ready'):
            return {'preferred_path': 1, 'reason': 'path2_warming_up', 'fallback_path': 1}

        q1, q2 = pred1['quality_score'], pred2['quality_score']
        c1, c2 = pred1['confidence'],    pred2['confidence']

        CONFIDENCE_GATE = 0.5

        p1_degraded = pred1['label'] == 'degraded' and c1 >= CONFIDENCE_GATE
        p2_degraded = pred2['label'] == 'degraded' and c2 >= CONFIDENCE_GATE

        if p1_degraded and not p2_degraded:
            return {'preferred_path': 2, 'reason': 'path1_predicted_degraded',  'fallback_path': 1}
        elif p2_degraded and not p1_degraded:
            return {'preferred_path': 1, 'reason': 'path2_predicted_degraded',  'fallback_path': 2}
        elif p1_degraded and p2_degraded:
            preferred = 1 if q1 >= q2 else 2
            return {'preferred_path': preferred, 'reason': 'both_degraded_choose_better', 'fallback_path': preferred}
        else:
            preferred = 1 if q1 >= q2 else 2
            return {'preferred_path': preferred, 'reason': 'quality_based', 'fallback_path': 2 if preferred == 1 else 1}