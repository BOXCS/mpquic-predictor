# evaluation/compare.py — FIXED VERSION
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import time
import random
import csv
from datetime import datetime
from model.predictor import PathPredictor

RESULT_DIR = 'evaluation/results fixed'
os.makedirs(RESULT_DIR, exist_ok=True)

PAYLOAD_BYTES = 312

TEST_SCENARIOS = [
    {
        'name':   'normal',
        'rounds': 100,
        'path1':  {'base_delay': 20,  'jitter': 5,  'loss': 1.0},
        'path2':  {'base_delay': 80,  'jitter': 20, 'loss': 5.0},
    },
    {
        'name':   'path1_degraded',
        'rounds': 100,
        'path1':  {'base_delay': 180, 'jitter': 40, 'loss': 20.0},
        'path2':  {'base_delay': 80,  'jitter': 20, 'loss': 5.0},
    },
    {
        'name':   'path2_degraded',
        'rounds': 100,
        'path1':  {'base_delay': 20,  'jitter': 5,  'loss': 1.0},
        'path2':  {'base_delay': 200, 'jitter': 50, 'loss': 25.0},
    },
    {
        'name':   'both_degraded',
        'rounds': 100,
        'path1':  {'base_delay': 150, 'jitter': 40, 'loss': 15.0},
        'path2':  {'base_delay': 200, 'jitter': 50, 'loss': 20.0},
    },
    # FIX: tambah skenario kritis — RTT tinggi tapi loss rendah
    # Ini yang sebelumnya gagal karena model lama tidak terlatih untuk ini
    {
        'name':   'path2_high_rtt_low_loss',
        'rounds': 100,
        'path1':  {'base_delay': 20,  'jitter': 5,  'loss': 1.0},
        'path2':  {'base_delay': 180, 'jitter': 40, 'loss': 3.0},
    },
    {
        'name':   'path1_high_rtt_low_loss',
        'rounds': 100,
        'path1':  {'base_delay': 160, 'jitter': 35, 'loss': 2.0},
        'path2':  {'base_delay': 80,  'jitter': 20, 'loss': 5.0},
    },
]


def simulate_network(base_delay, jitter, loss_pct):
    """Simulasi kondisi jaringan, return (rtt, throughput, status)"""
    if random.random() < loss_pct / 100:
        return None, 0, 'dropped'
    delay = max(1, base_delay + random.uniform(-jitter, jitter))
    time.sleep(delay / 1000)
    # FIX: throughput dinamis, bukan hardcode 312
    throughput = round((PAYLOAD_BYTES * (1 - loss_pct/100)) / (delay/1000), 2)
    return round(delay, 2), throughput, 'success'


def round_robin_test(scenario):
    results      = []
    current_path = 1
    for i in range(scenario['rounds']):
        cond = scenario[f'path{current_path}']
        rtt, throughput, status = simulate_network(
            cond['base_delay'], cond['jitter'], cond['loss']
        )
        results.append({
            'round':       i + 1,
            'method':      'round_robin',
            'scenario':    scenario['name'],
            'path_chosen': current_path,
            'rtt_ms':      rtt or 0,
            'status':      status,
            'reason':      'round_robin',
            'timestamp':   datetime.now().isoformat()
        })
        current_path = 2 if current_path == 1 else 1
    return results


def lstm_test(scenario, predictor):
    results = []

    # Warm up buffer
    print(f"    Warming up buffer ({predictor.window} records)...")
    for _ in range(predictor.window + 5):
        for pid, pk in [(1, 'path1'), (2, 'path2')]:
            cond = scenario[pk]
            rtt, throughput, status = simulate_network(
                cond['base_delay'], cond['jitter'], cond['loss']
            )
            # FIX: tidak ada packet_loss_pct, throughput dinamis
            predictor.add_record(
                pid,
                rtt or 0,
                throughput,
                status
            )
        time.sleep(0.02)

    for i in range(scenario['rounds']):
        rtt1, tp1, status1 = simulate_network(
            scenario['path1']['base_delay'],
            scenario['path1']['jitter'],
            scenario['path1']['loss']
        )
        rtt2, tp2, status2 = simulate_network(
            scenario['path2']['base_delay'],
            scenario['path2']['jitter'],
            scenario['path2']['loss']
        )

        # FIX: tidak ada packet_loss_pct, throughput dinamis
        predictor.add_record(1, rtt1 or 0, tp1, status1)
        predictor.add_record(2, rtt2 or 0, tp2, status2)

        pred1 = predictor.predict(1)
        pred2 = predictor.predict(2)
        rec   = predictor.get_recommendation(pred1, pred2)
        chosen_path = rec['preferred_path']

        chosen_rtt    = rtt1    if chosen_path == 1 else rtt2
        chosen_status = status1 if chosen_path == 1 else status2

        results.append({
            'round':       i + 1,
            'method':      'lstm_predictor',
            'scenario':    scenario['name'],
            'path_chosen': chosen_path,
            'rtt_ms':      chosen_rtt or 0,
            'status':      chosen_status,
            'reason':      rec['reason'],
            'timestamp':   datetime.now().isoformat()
        })
        time.sleep(0.02)

    return results


def analyze(results, method_name):
    total   = len(results)
    success = [r for r in results if r['status'] == 'success']
    dropped = [r for r in results if r['status'] == 'dropped']
    rtts    = [r['rtt_ms'] for r in success]

    avg_rtt      = sum(rtts) / len(rtts) if rtts else 0
    loss_rate    = len(dropped) / total * 100
    success_rate = len(success) / total * 100

    print(f"\n  [{method_name}]")
    print(f"    Success rate : {success_rate:.1f}% ({len(success)}/{total})")
    print(f"    Packet loss  : {loss_rate:.1f}%")
    print(f"    Avg RTT      : {avg_rtt:.1f}ms")
    if rtts:
        print(f"    Min RTT      : {min(rtts):.1f}ms")
        print(f"    Max RTT      : {max(rtts):.1f}ms")

    return {
        'method':       method_name,
        'success_rate': round(success_rate, 2),
        'loss_rate':    round(loss_rate, 2),
        'avg_rtt':      round(avg_rtt, 2),
        'total_rounds': total
    }


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

        print("\n  Menjalankan Round Robin...")
        rr_results  = round_robin_test(scenario)
        rr_summary  = analyze(rr_results, 'Round Robin')

        print("\n  Menjalankan LSTM Predictor...")
        lstm_results  = lstm_test(scenario, predictor)
        lstm_summary  = analyze(lstm_results, 'LSTM Predictor')

        rtt_diff  = rr_summary['avg_rtt']   - lstm_summary['avg_rtt']
        loss_diff = rr_summary['loss_rate'] - lstm_summary['loss_rate']

        print(f"\n  [IMPROVEMENT]")
        print(f"    RTT  : {rtt_diff:+.1f}ms "
              f"({'lebih baik' if rtt_diff > 0 else 'lebih buruk'})")
        print(f"    Loss : {loss_diff:+.1f}% "
              f"({'lebih baik' if loss_diff > 0 else 'lebih buruk'})")

        rr_summary['scenario']   = scenario['name']
        lstm_summary['scenario'] = scenario['name']
        summary.append(rr_summary)
        summary.append(lstm_summary)
        all_results.extend(rr_results + lstm_results)

    result_file  = os.path.join(RESULT_DIR, 'comparison_results.csv')
    summary_file = os.path.join(RESULT_DIR, 'summary.csv')

    with open(result_file, 'w', newline='') as f:
        fieldnames = ['round', 'method', 'scenario', 'path_chosen',
                      'rtt_ms', 'status', 'reason', 'timestamp']
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_results)

    with open(summary_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    print(f"\n{'='*50}")
    print(f"Hasil lengkap : {result_file}")
    print(f"Ringkasan     : {summary_file}")
    print("\n=== Evaluasi selesai ===")