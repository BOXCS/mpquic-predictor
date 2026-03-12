# model/train.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import os

# ── Config ──────────────────────────────────────────────
DATA_PATH   = 'data/logs/path_log.csv'
MODEL_DIR   = 'model/saved'
WINDOW_SIZE = 20      # 20 records terakhir sebagai input LSTM
HORIZON     = 5       # prediksi 5 langkah ke depan
THRESHOLD   = 100     # RTT > 100ms dianggap degradasi
os.makedirs(MODEL_DIR, exist_ok=True)

# ── 1. Load & basic clean ────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

# Cap outlier RTT
df['rtt_ms'] = df['rtt_ms'].clip(upper=500)

# Encode status
df['status_enc'] = (df['status'] == 'success').astype(int)

print(f"Total records: {len(df)}")
print(df[['path_id','rtt_ms','packet_loss_pct','status']].describe())

# ── 2. Feature engineering per path ─────────────────────
def build_features(df_path):
    d = df_path.copy().reset_index(drop=True)

    # Rolling stats (window 5)
    d['rtt_roll_mean'] = d['rtt_ms'].rolling(5, min_periods=1).mean()
    d['rtt_roll_std']  = d['rtt_ms'].rolling(5, min_periods=1).std().fillna(0)
    d['rtt_diff']      = d['rtt_ms'].diff().fillna(0)

    # Label: apakah dalam HORIZON langkah ke depan ada degradasi?
    d['degraded'] = (d['rtt_ms'] > THRESHOLD).astype(int)
    d['label'] = 0
    for i in range(len(d) - HORIZON):
        if d['degraded'].iloc[i+1 : i+1+HORIZON].any():
            d.at[i, 'label'] = 1

    return d

df1 = build_features(df[df['path_id'] == 1])
df2 = build_features(df[df['path_id'] == 2])
df_all = pd.concat([df1, df2]).sort_values('timestamp').reset_index(drop=True)

print(f"\nLabel distribution:")
print(df_all['label'].value_counts())

# ── 3. Build sequences ───────────────────────────────────
FEATURES = ['rtt_ms', 'rtt_roll_mean', 'rtt_roll_std',
            'rtt_diff', 'packet_loss_pct', 'status_enc']

scaler = MinMaxScaler()
df_all[FEATURES] = scaler.fit_transform(df_all[FEATURES])

def make_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[FEATURES].iloc[i:i+window].values)
        y.append(data['label'].iloc[i+window])
    return np.array(X), np.array(y)

X, y = make_sequences(df_all, WINDOW_SIZE)
print(f"\nSequences: X={X.shape}, y={y.shape}")
print(f"Class balance: {y.mean():.2%} degradation")

# ── 4. Train/test split ──────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

# ── 5. Build model ───────────────────────────────────────
model = Sequential([
    LSTM(64, input_shape=(WINDOW_SIZE, len(FEATURES)),
         return_sequences=True),
    Dropout(0.3),
    LSTM(32, return_sequences=False),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)
model.summary()

# ── 6. Train dengan class weights ────────────────────────
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

# ── 7. Evaluate ──────────────────────────────────────────
print("\n=== Evaluation ===")
y_pred = (model.predict(X_test) > 0.3).astype(int).flatten()
print(classification_report(y_test, y_pred,
      target_names=['stable', 'degraded']))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ── 8. Save ──────────────────────────────────────────────
model.save(os.path.join(MODEL_DIR, 'lstm_model.keras'))
with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
with open(os.path.join(MODEL_DIR, 'config.pkl'), 'wb') as f:
    pickle.dump({
        'window_size': WINDOW_SIZE,
        'horizon':     HORIZON,
        'threshold':   THRESHOLD,
        'features':    FEATURES
    }, f)

print(f"\nModel saved to {MODEL_DIR}/")
print("Files: lstm_model.keras, scaler.pkl, config.pkl")

# ── 9. Threshold analysis ────────────────────────────────
from sklearn.metrics import precision_recall_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

y_prob = model.predict(X_test).flatten()
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

# Cari threshold dengan F1 terbaik
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_idx = f1_scores.argmax()
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

print(f"\n=== Threshold Analysis ===")
print(f"Best threshold: {best_threshold:.3f}")
print(f"Best F1: {best_f1:.3f}")
print(f"At best threshold — Precision: {precisions[best_idx]:.3f} | Recall: {recalls[best_idx]:.3f}")

# Tampilkan beberapa threshold kandidat
print(f"\nThreshold candidates:")
print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
for t, p, r in zip(thresholds[::len(thresholds)//10],
                    precisions[::len(thresholds)//10],
                    recalls[::len(thresholds)//10]):
    f1 = 2*p*r/(p+r+1e-8)
    print(f"{t:>10.3f} {p:>10.3f} {r:>10.3f} {f1:>10.3f}")

# Plot precision-recall curve
plt.figure(figsize=(8,5))
plt.plot(recalls, precisions, 'b-', linewidth=2)
plt.axvline(x=recalls[best_idx], color='r', linestyle='--',
            label=f'Best threshold={best_threshold:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve — LSTM Degradation Predictor')
plt.legend()
plt.grid(True)
plt.savefig('model/saved/precision_recall_curve.png', dpi=150)
print(f"\nCurve saved to model/saved/precision_recall_curve.png")

# Final eval dengan best threshold
print(f"\n=== Final Evaluation (threshold={best_threshold:.3f}) ===")
y_pred_best = (y_prob > best_threshold).astype(int)
print(classification_report(y_test, y_pred_best,
      target_names=['stable', 'degraded']))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_best))

# Update config dengan best threshold
with open(os.path.join(MODEL_DIR, 'config.pkl'), 'wb') as f:
    pickle.dump({
        'window_size': WINDOW_SIZE,
        'horizon':     HORIZON,
        'threshold_rtt':   THRESHOLD,
        'pred_threshold':  best_threshold,
        'features':    FEATURES
    }, f)
print(f"\nConfig updated dengan best threshold: {best_threshold:.3f}")