# evaluation/test_prediction.py
# Menguji kemampuan LSTM memprediksi degradasi sebelum threshold tercapai
# Dua skenario: gradual (RTT naik bertahap) dan sudden (RTT naik tiba-tiba)

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import time
import random
import csv
from datetime import datetime
from model.predictor import PathPredictor

RESULT_DIR = 'evaluation/results'
os.makedirs(RESULT_DIR, exist_ok=True)

PAYLOAD_BYTES = 312
THRESHOLD_RTT = 100  # ms — ambang degradasi yang dipakai saat training


def simulate_rtt(base_delay, jitter):
    """Simulasi RTT dengan jitter"""
    delay = max(1, base_delay + random.uniform(-jitter, jitter))
    return round(delay, 2)


def calculate_throughput(delay_ms, loss_pct):
    delay_s = delay_ms / 1000
    if delay_s <= 0:
        return 0
    return round((PAYLOAD_BYTES * (1 - loss_pct / 100)) / delay_s, 2)


def run_test(scenario_name, steps, predictor, path_id=1):
    """
    Jalankan satu skenario pengujian.
    steps: list of dict {rtt_base, jitter, loss, label}
      label: 'normal' | 'degrading' | 'degraded'
    Kembalikan list hasil per timestep.
    """
    results = []
    step_num = 0

    # Warm up buffer dulu dengan kondisi normal
    print(f"\n  Warming up ({predictor.window} records)...")
    for _ in range(predictor.window + 5):
        rtt = simulate_rtt(20, 5)
        tp  = calculate_throughput(rtt, 1.0)
        predictor.add_record(path_id, rtt, tp, 'success')

    print(f"  Menjalankan skenario '{scenario_name}'...")
    for step in steps:
        rtt    = simulate_rtt(step['rtt_base'], step['jitter'])
        loss   = step['loss']
        status = 'dropped' if random.random() < loss / 100 else 'success'
        tp     = calculate_throughput(rtt, loss) if status == 'success' else 0

        predictor.add_record(path_id, rtt if status == 'success' else 0, tp, status)
        pred = predictor.predict(path_id)

        already_degraded = rtt >= THRESHOLD_RTT

        row = {
            'scenario':          scenario_name,
            'step':              step_num,
            'ground_truth':      step['label'],
            'rtt_ms':            rtt,
            'status':            status,
            'already_degraded':  already_degraded,
            'pred_ready':        pred.get('ready', False),
            'pred_label':        pred.get('label', 'warming_up'),
            'degradation_prob':  pred.get('degradation_probability', 0),
            'quality_score':     pred.get('quality_score', 1),
            'confidence':        pred.get('confidence', 0),
            'timestamp':         datetime.now().isoformat()
        }
        results.append(row)

        # Print ringkas ke terminal
        if pred.get('ready'):
            prob   = pred['degradation_probability']
            label  = pred['label']
            marker = ''
            if label == 'degraded' and not already_degraded:
                marker = ' <<< EARLY WARNING'
            elif label == 'degraded' and already_degraded:
                marker = ' [sudah degradasi]'
            elif label == 'degrading':
                marker = ' [mulai degrading]'

            print(f"    Step {step_num:3d} | RTT={rtt:6.1f}ms | "
                  f"GT={step['label']:9s} | pred={label:9s} | "
                  f"prob={prob:.3f}{marker}")

        step_num += 1

    return results


# ── SKENARIO 1: GRADUAL DEGRADATION ─────────────────────────────────────────
# RTT naik perlahan dari 20ms ke 200ms selama 60 steps
# LSTM seharusnya mulai warning SEBELUM RTT menyentuh 100ms
def build_gradual_steps():
    steps = []
    total = 60
    for i in range(total):
        progress = i / (total - 1)
        rtt_base = 20 + progress * 180   # 20ms → 200ms
        jitter   = 5  + progress * 20    # jitter ikut naik
        loss     = 0.5 + progress * 4.5  # loss 0.5% → 5%

        if rtt_base < 70:
            label = 'normal'
        elif rtt_base < THRESHOLD_RTT:
            label = 'degrading'
        else:
            label = 'degraded'

        steps.append({
            'rtt_base': round(rtt_base, 1),
            'jitter':   round(jitter, 1),
            'loss':     round(loss, 2),
            'label':    label
        })
    return steps


# ── SKENARIO 2: SUDDEN DEGRADATION ──────────────────────────────────────────
# RTT stabil di 20ms lalu tiba-tiba loncat ke 180ms di step ke-30
# LSTM seharusnya bereaksi cepat setelah beberapa records pertama degradasi
def build_sudden_steps():
    steps = []
    for i in range(60):
        if i < 30:
            steps.append({'rtt_base': 20,  'jitter': 5,  'loss': 1.0,  'label': 'normal'})
        else:
            steps.append({'rtt_base': 180, 'jitter': 30, 'loss': 15.0, 'label': 'degraded'})
    return steps


# ── SKENARIO 3: DEGRADATION LALU RECOVERY ────────────────────────────────────
# RTT naik ke 180ms lalu turun kembali ke normal
# LSTM seharusnya berhenti warning setelah recovery
def build_recovery_steps():
    steps = []
    for i in range(80):
        if i < 20:
            steps.append({'rtt_base': 20,  'jitter': 5,  'loss': 1.0,  'label': 'normal'})
        elif i < 40:
            steps.append({'rtt_base': 180, 'jitter': 30, 'loss': 15.0, 'label': 'degraded'})
        else:
            steps.append({'rtt_base': 20,  'jitter': 5,  'loss': 1.0,  'label': 'normal'})
    return steps


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=== Pengujian Prediksi Degradasi LSTM ===")
    print(f"Threshold RTT : {THRESHOLD_RTT}ms")
    print(f"Tujuan        : LSTM harus warning SEBELUM RTT menyentuh {THRESHOLD_RTT}ms\n")

    predictor = PathPredictor()
    all_results = []

    # ── Skenario 1: Gradual
    print("\n" + "="*55)
    print("SKENARIO 1: Gradual Degradation (RTT naik perlahan)")
    print("="*55)
    predictor.buffers = {1: [], 2: []}   # reset buffer antar skenario
    results_gradual = run_test(
        'gradual_degradation',
        build_gradual_steps(),
        predictor, path_id=1
    )
    all_results.extend(results_gradual)

    # ── Skenario 2: Sudden
    print("\n" + "="*55)
    print("SKENARIO 2: Sudden Degradation (RTT loncat tiba-tiba)")
    print("="*55)
    predictor.buffers = {1: [], 2: []}
    results_sudden = run_test(
        'sudden_degradation',
        build_sudden_steps(),
        predictor, path_id=1
    )
    all_results.extend(results_sudden)

    # ── Skenario 3: Recovery
    print("\n" + "="*55)
    print("SKENARIO 3: Degradation + Recovery")
    print("="*55)
    predictor.buffers = {1: [], 2: []}
    results_recovery = run_test(
        'degradation_recovery',
        build_recovery_steps(),
        predictor, path_id=1
    )
    all_results.extend(results_recovery)

    # ── Simpan CSV
    out_file = os.path.join(RESULT_DIR, 'prediction_test.csv')
    with open(out_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nHasil disimpan ke: {out_file}")

    # ── Ringkasan per skenario
    print("\n=== Ringkasan ===")
    for scenario_name in ['gradual_degradation', 'sudden_degradation', 'degradation_recovery']:
        rows = [r for r in all_results if r['scenario'] == scenario_name]
        ready = [r for r in rows if r['pred_ready']]

        # Early warning: LSTM prediksi degraded SEBELUM RTT mencapai threshold
        early_warn = [r for r in ready
                      if r['pred_label'] == 'degraded'
                      and not r['already_degraded']]

        # True positive: LSTM prediksi degraded saat RTT memang sudah degraded
        tp = [r for r in ready
              if r['pred_label'] == 'degraded'
              and r['ground_truth'] == 'degraded']

        # False positive: LSTM prediksi degraded saat kondisi normal
        fp = [r for r in ready
              if r['pred_label'] == 'degraded'
              and r['ground_truth'] == 'normal']

        # False negative: LSTM tidak prediksi degraded saat kondisi degraded
        fn = [r for r in ready
              if r['pred_label'] != 'degraded'
              and r['ground_truth'] == 'degraded']

        print(f"\n  {scenario_name}:")
        print(f"    Early warning (pred degraded sebelum RTT > {THRESHOLD_RTT}ms) : {len(early_warn)} steps")
        print(f"    True positive  : {len(tp)}")
        print(f"    False positive : {len(fp)}")
        print(f"    False negative : {len(fn)}")

        if early_warn:
            first = early_warn[0]
            print(f"    Pertama kali warning di step {first['step']} "
                  f"(RTT={first['rtt_ms']}ms, prob={first['degradation_prob']:.3f})")

    print("\n=== Selesai ===")