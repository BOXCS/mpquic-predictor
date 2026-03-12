# evaluation/compare.py
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import requests
import time
import random
import csv
from datetime import datetime
from model.predictor import PathPredictor

RESULT_DIR = 'evaluation/results'
os.makedirs(RESULT_DIR, exist_ok=True)

# ── Simulasi kondisi jaringan ────────────────────────────
def simulate_network(base_delay, jitter, loss_pct):
    if random.random() < loss_pct / 100:
        return None, 'dropped'
    delay = base_delay + random.uniform(-jitter, jitter)
    delay = max(1, delay)
    time.sleep(delay / 1000)
    return round(delay, 2), 'success'

# ── Skenario pengujian ───────────────────────────────────
TEST_SCENARIOS = [
    {
        'name': 'normal',
        'rounds': 100,
        'path1': {'base_delay': 20,  'jitter': 5,  'loss': 1.0},
        'path2': {'base_delay': 80,  'jitter': 20, 'loss': 5.0},
    },
    {
        'name': 'path1_degraded',
        'rounds': 100,
        'path1': {'base_delay': 180, 'jitter': 40, 'loss': 20.0},
        'path2': {'base_delay': 80,  'jitter': 20, 'loss': 5.0},
    },
    {
        'name': 'path2_degraded',
        'rounds': 100,
        'path1': {'base_delay': 20,  'jitter': 5,  'loss': 1.0},
        'path2': {'base_delay': 200, 'jitter': 50, 'loss': 25.0},
    },
    {
        'name': 'both_degraded',
        'rounds': 100,
        'path1': {'base_delay': 150, 'jitter': 40, 'loss': 15.0},
        'path2': {'base_delay': 200, 'jitter': 50, 'loss': 20.0},
    },
]

# ── Round Robin baseline ─────────────────────────────────
def round_robin_test(scenario):
    results = []
    current_path = 1

    for i in range(scenario['rounds']):
        cond = scenario[f'path{current_path}']
        rtt, status = simulate_network(
            cond['base_delay'], cond['jitter'], cond['loss']
        )
        results.append({
            'round':        i + 1,
            'method':       'round_robin',
            'scenario':     scenario['name'],
            'path_chosen':  current_path,
            'rtt_ms':       rtt if rtt else 0,
            'status':       status,
            'timestamp':    datetime.now().isoformat()
        })
        # Bergantian tanpa mempertimbangkan kondisi
        current_path = 2 if current_path == 1 else 1

    return results

# ── LSTM predictor test ──────────────────────────────────
def lstm_test(scenario, predictor):
    results = []

    # Warm up buffer dulu
    print(f"    Warming up buffer ({predictor.window} records)...")
    for _ in range(predictor.window + 5):
        for pid, pk in [(1, 'path1'), (2, 'path2')]:
            cond = scenario[pk]
            rtt, status = simulate_network(
                cond['base_delay'], cond['jitter'], cond['loss']
            )
            predictor.add_record(
                pid,
                rtt if rtt else 0,
                312,
                cond['loss'],
                status
            )
        time.sleep(0.05)

    for i in range(scenario['rounds']):
        # Simulasi kondisi kedua jalur
        rtt1, status1 = simulate_network(
            scenario['path1']['base_delay'],
            scenario['path1']['jitter'],
            scenario['path1']['loss']
        )
        rtt2, status2 = simulate_network(
            scenario['path2']['base_delay'],
            scenario['path2']['jitter'],
            scenario['path2']['loss']
        )

        # Update predictor
        predictor.add_record(1, rtt1 or 0, 312, scenario['path1']['loss'], status1)
        predictor.add_record(2, rtt2 or 0, 312, scenario['path2']['loss'], status2)

        # Ambil rekomendasi
        pred1 = predictor.predict(1)
        pred2 = predictor.predict(2)
        rec   = predictor.get_recommendation(pred1, pred2)
        chosen_path = rec['preferred_path']

        # Ambil hasil jalur yang dipilih
        chosen_rtt    = rtt1    if chosen_path == 1 else rtt2
        chosen_status = status1 if chosen_path == 1 else status2

        results.append({
            'round':        i + 1,
            'method':       'lstm_predictor',
            'scenario':     scenario['name'],
            'path_chosen':  chosen_path,
            'rtt_ms':       chosen_rtt if chosen_rtt else 0,
            'status':       chosen_status,
            'reason':       rec['reason'],
            'timestamp':    datetime.now().isoformat()
        })
        time.sleep(0.05)

    return results

# ── Analisis hasil ───────────────────────────────────────
def analyze(results, method_name):
    total       = len(results)
    success     = [r for r in results if r['status'] == 'success']
    dropped     = [r for r in results if r['status'] == 'dropped']
    rtts        = [r['rtt_ms'] for r in success]

    avg_rtt     = sum(rtts) / len(rtts) if rtts else 0
    loss_rate   = len(dropped) / total * 100
    success_rate= len(success) / total * 100

    print(f"\n  [{method_name}]")
    print(f"    Success rate : {success_rate:.1f}% ({len(success)}/{total})")
    print(f"    Packet loss  : {loss_rate:.1f}%")
    print(f"    Avg RTT      : {avg_rtt:.1f}ms")
    print(f"    Min RTT      : {min(rtts):.1f}ms" if rtts else "    Min RTT: N/A")
    print(f"    Max RTT      : {max(rtts):.1f}ms" if rtts else "    Max RTT: N/A")

    return {
        'method':       method_name,
        'success_rate': round(success_rate, 2),
        'loss_rate':    round(loss_rate, 2),
        'avg_rtt':      round(avg_rtt, 2),
        'total_rounds': total
    }

# ── Main ─────────────────────────────────────────────────
if __name__ == '__main__':
    print("=== Evaluasi Komparatif: Round Robin vs LSTM Predictor ===\n")

    predictor   = PathPredictor()
    all_results = []
    summary     = []

    for scenario in TEST_SCENARIOS:
        print(f"\n{'='*50}")
        print(f"Skenario: {scenario['name'].upper()}")
        print(f"Path1: delay={scenario['path1']['base_delay']}ms "
              f"loss={scenario['path1']['loss']}%")
        print(f"Path2: delay={scenario['path2']['base_delay']}ms "
              f"loss={scenario['path2']['loss']}%")

        # Round Robin
        print("\n  Menjalankan Round Robin...")
        rr_results = round_robin_test(scenario)
        rr_summary = analyze(rr_results, 'Round Robin')

        # LSTM
        print("\n  Menjalankan LSTM Predictor...")
        lstm_results = lstm_test(scenario, predictor)
        lstm_summary = analyze(lstm_results, 'LSTM Predictor')

        # Perbandingan
        rtt_improvement  = rr_summary['avg_rtt'] - lstm_summary['avg_rtt']
        loss_improvement = rr_summary['loss_rate'] - lstm_summary['loss_rate']

        print(f"\n  [IMPROVEMENT]")
        print(f"    RTT  : {rtt_improvement:+.1f}ms "
              f"({'lebih baik' if rtt_improvement > 0 else 'lebih buruk'})")
        print(f"    Loss : {loss_improvement:+.1f}% "
              f"({'lebih baik' if loss_improvement > 0 else 'lebih buruk'})")

        rr_summary['scenario']   = scenario['name']
        lstm_summary['scenario'] = scenario['name']
        summary.append(rr_summary)
        summary.append(lstm_summary)
        all_results.extend(rr_results + lstm_results)

    # Simpan ke CSV
    result_file  = os.path.join(RESULT_DIR, 'comparison_results.csv')
    summary_file = os.path.join(RESULT_DIR, 'summary.csv')

    # Ganti bagian simpan ke CSV
    with open(result_file, 'w', newline='') as f:
        fieldnames = ['round', 'method', 'scenario', 'path_chosen',
                      'rtt_ms', 'status', 'reason', 'timestamp']
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        # Tambahkan field reason ke RR results yang tidak punya
        for r in all_results:
            if 'reason' not in r:
                r['reason'] = 'round_robin'
        writer.writerows(all_results)

    with open(summary_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    print(f"\n{'='*50}")
    print(f"Hasil lengkap : {result_file}")
    print(f"Ringkasan     : {summary_file}")
    print("\n=== Evaluasi selesai ===")