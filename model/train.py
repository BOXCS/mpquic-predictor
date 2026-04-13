# model/train.py — FINAL FIXED VERSION
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import os

# ── Config ───────────────────────────────────────────────
DATA_PATH   = 'data/logs/path_log.csv'
MODEL_DIR   = 'model/saved/rtt only'
WINDOW_SIZE = 20
HORIZON     = 5
THRESHOLD   = 100
os.makedirs(MODEL_DIR, exist_ok=True)

# FIX: throughput_bps ditambahkan ke FEATURES
# sebelumnya tidak dipakai padahal sudah ada di data dan sudah dinamis
FEATURES = ['rtt_ms', 'rtt_roll_mean', 'rtt_roll_std',
            'rtt_diff', 'throughput_bps', 'status_enc']

# ── 1. Load & clean ──────────────────────────────────────
print("Loading data...")
df = pd.read_csv(
    DATA_PATH,
    names=['timestamp', 'path_id', 'rtt_ms', 'throughput_bps',
           'packet_loss_pct', 'status'],
    parse_dates=['timestamp']
)
df = df.sort_values('timestamp').reset_index(drop=True)

# Hapus records dropped (rtt=0) — ini noise, bukan sinyal
# Dropped packets dicatat dengan rtt=0 yang akan membingungkan LSTM
df = df[df['rtt_ms'] > 0].reset_index(drop=True)

df['rtt_ms'] = df['rtt_ms'].clip(upper=500)
df['throughput_bps'] = df['throughput_bps'].clip(upper=50000)
df['status_enc'] = (df['status'] == 'success').astype(int)

print(f"Total records setelah filter: {len(df)}")
print(df[['path_id', 'rtt_ms', 'throughput_bps', 'packet_loss_pct']].describe())

# ── 2. Feature engineering per path ─────────────────────
def build_features(df_path):
    d = df_path.copy().reset_index(drop=True)
    d['rtt_roll_mean'] = d['rtt_ms'].rolling(5, min_periods=1).mean()
    d['rtt_roll_std']  = d['rtt_ms'].rolling(5, min_periods=1).std().fillna(0)
    d['rtt_diff']      = d['rtt_ms'].diff().fillna(0)
    d['degraded']      = (d['rtt_ms'] > THRESHOLD).astype(int)
    d['label'] = 0
    for i in range(len(d) - HORIZON):
        if d['degraded'].iloc[i+1 : i+1+HORIZON].any():
            d.at[i, 'label'] = 1
    return d

df1 = build_features(df[df['path_id'] == 1])
df2 = build_features(df[df['path_id'] == 2])

print(f"\nLabel distribution path1: {df1['label'].value_counts().to_dict()}")
print(f"Label distribution path2: {df2['label'].value_counts().to_dict()}")

# ── 3. Scaler ────────────────────────────────────────────
df_all_for_scaler = pd.concat([df1, df2])
scaler = MinMaxScaler()
scaler.fit(df_all_for_scaler[FEATURES])
df1[FEATURES] = scaler.transform(df1[FEATURES])
df2[FEATURES] = scaler.transform(df2[FEATURES])

# ── 4. Build sequences per path ──────────────────────────
def make_sequences(data, window):
    X, y = [], []
    vals   = data[FEATURES].values
    labels = data['label'].values
    for i in range(len(data) - window):
        X.append(vals[i:i+window])
        y.append(labels[i+window])
    return np.array(X), np.array(y)

X1, y1 = make_sequences(df1, WINDOW_SIZE)
X2, y2 = make_sequences(df2, WINDOW_SIZE)

X = np.concatenate([X1, X2], axis=0)
y = np.concatenate([y1, y2], axis=0)

print(f"\nTotal sequences: X={X.shape}, y={y.shape}")
print(f"Class balance sebelum split: {y.mean():.2%} degradation")

# ── 5. Shuffle lalu split ────────────────────────────────
# FIX UTAMA: setelah concat, semua sequence Path1 ada di depan
# dan Path2 di belakang. Kalau langsung split 80/20, test set
# akan didominasi Path2 yang distribusinya berbeda.
# Shuffle dulu dengan seed tetap supaya reproducible,
# baru split — ini yang sebelumnya belum ada di kode.
rng = np.random.default_rng(seed=42)
idx = rng.permutation(len(X))
X, y = X[idx], y[idx]

split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"\nTrain : {len(X_train)} sequences")
print(f"Test  : {len(X_test)} sequences")
print(f"Train degradation : {y_train.mean():.2%}")
print(f"Test degradation  : {y_test.mean():.2%}")
# Kedua angka di atas harus mirip — kalau jauh berbeda, ada masalah distribusi

# ── 6. Model ─────────────────────────────────────────────
model = Sequential([
    LSTM(64, input_shape=(WINDOW_SIZE, len(FEATURES)), return_sequences=True),
    Dropout(0.3),
    LSTM(32, return_sequences=False),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1,  activation='sigmoid')
])
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)
model.summary()

# ── 7. Class weights & training ──────────────────────────
from sklearn.utils.class_weight import compute_class_weight

classes = np.unique(y_train)
weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight = dict(zip(classes, weights))
print(f"\nClass weights: {class_weight}")

early_stop = EarlyStopping(
    monitor='val_auc', patience=15,
    restore_best_weights=True, mode='max'
)

print("\nTraining...")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    class_weight=class_weight,
    verbose=1
)

# ── 8. Threshold analysis ────────────────────────────────
from sklearn.metrics import precision_recall_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

y_prob = model.predict(X_test).flatten()
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_idx       = f1_scores.argmax()
best_threshold = float(thresholds[best_idx])
best_f1        = f1_scores[best_idx]

print(f"\n=== Threshold Analysis ===")
print(f"Best threshold : {best_threshold:.3f}")
print(f"Best F1        : {best_f1:.3f}")
print(f"Precision      : {precisions[best_idx]:.3f}")
print(f"Recall         : {recalls[best_idx]:.3f}")

plt.figure(figsize=(8, 5))
plt.plot(recalls, precisions, 'b-', linewidth=2)
plt.axvline(x=recalls[best_idx], color='r', linestyle='--',
            label=f'Best threshold={best_threshold:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve — LSTM Degradation Predictor')
plt.legend()
plt.grid(True)
plt.savefig('model/saved/rtt only/precision_recall_curve.png', dpi=150)
print("Curve saved to model/saved/rtt only/precision_recall_curve.png")

# ── 9. Final evaluation ──────────────────────────────────
print(f"\n=== Final Evaluation (threshold={best_threshold:.3f}) ===")
y_pred = (y_prob > best_threshold).astype(int)
print(classification_report(y_test, y_pred,
      target_names=['stable', 'degraded']))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ── 10. Save ─────────────────────────────────────────────
model.save(os.path.join(MODEL_DIR, 'lstm_model.keras'))
with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
with open(os.path.join(MODEL_DIR, 'config.pkl'), 'wb') as f:
    pickle.dump({
        'window_size'    : WINDOW_SIZE,
        'horizon'        : HORIZON,
        'threshold_rtt'  : THRESHOLD,
        'pred_threshold' : best_threshold,
        'features'       : FEATURES
    }, f)
print(f"\nModel saved to {MODEL_DIR}/")
print(f"pred_threshold tersimpan: {best_threshold:.3f}")