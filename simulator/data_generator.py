# simulator/data_generator.py
import requests
import time
import random
import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from logger.path_logger import log_entry

# Skenario kondisi jaringan yang akan disimulasikan
SCENARIOS = [
    {
        'name': 'normal',
        'duration_seconds': 120,
        'path1': {'base_delay': 20,  'jitter': 5,  'loss': 1.0},
        'path2': {'base_delay': 80,  'jitter': 20, 'loss': 5.0},
    },
    {
        'name': 'path1_degrading',
        'duration_seconds': 120,
        'path1': {'base_delay': 60,  'jitter': 20, 'loss': 8.0},
        'path2': {'base_delay': 80,  'jitter': 20, 'loss': 5.0},
    },
    {
        'name': 'path1_degraded',
        'duration_seconds': 120,
        'path1': {'base_delay': 150, 'jitter': 40, 'loss': 20.0},
        'path2': {'base_delay': 80,  'jitter': 20, 'loss': 5.0},
    },
    {
        'name': 'path2_degrading',
        'duration_seconds': 120,
        'path1': {'base_delay': 20,  'jitter': 5,  'loss': 1.0},
        'path2': {'base_delay': 120, 'jitter': 35, 'loss': 12.0},
    },
    {
        'name': 'path2_degraded',
        'duration_seconds': 120,
        'path1': {'base_delay': 20,  'jitter': 5,  'loss': 1.0},
        'path2': {'base_delay': 200, 'jitter': 50, 'loss': 25.0},
    },
    {
        'name': 'both_congested',
        'duration_seconds': 120,
        'path1': {'base_delay': 100, 'jitter': 30, 'loss': 15.0},
        'path2': {'base_delay': 180, 'jitter': 50, 'loss': 20.0},
    },
    {
        'name': 'both_congested_severe',
        'duration_seconds': 120,
        'path1': {'base_delay': 200, 'jitter': 60, 'loss': 30.0},
        'path2': {'base_delay': 250, 'jitter': 70, 'loss': 35.0},
    },
    {
        'name': 'recovery_path1',
        'duration_seconds': 120,
        'path1': {'base_delay': 25,  'jitter': 8,  'loss': 2.0},
        'path2': {'base_delay': 90,  'jitter': 25, 'loss': 6.0},
    },
    {
        'name': 'recovery_path2',
        'duration_seconds': 120,
        'path1': {'base_delay': 20,  'jitter': 5,  'loss': 1.0},
        'path2': {'base_delay': 85,  'jitter': 22, 'loss': 5.0},
    },
    {
        'name': 'normal_stable',
        'duration_seconds': 120,
        'path1': {'base_delay': 18,  'jitter': 3,  'loss': 0.5},
        'path2': {'base_delay': 75,  'jitter': 15, 'loss': 3.0},
    },
    # Tambahkan ke SCENARIOS yang sudah ada
    {
        'name': 'gradual_degradation_path1',
        'duration_seconds': 180,
        'path1': {'base_delay': 20,  'jitter': 5,  'loss': 1.0},
        'path2': {'base_delay': 80,  'jitter': 20, 'loss': 5.0},
        'transition': {
            'path': 1,
            'start_at': 60,
            'end_at': 120,
            'target_delay': 180,
            'target_loss': 25.0
        }
    },
    {
        'name': 'gradual_degradation_path2',
        'duration_seconds': 180,
        'path1': {'base_delay': 20,  'jitter': 5,  'loss': 1.0},
        'path2': {'base_delay': 80,  'jitter': 20, 'loss': 5.0},
        'transition': {
            'path': 2,
            'start_at': 60,
            'end_at': 120,
            'target_delay': 220,
            'target_loss': 30.0
        }
    },
]

def send_with_condition(path_id, port, condition):
    """Kirim request dengan kondisi jaringan yang disimulasikan"""
    # Simulasi packet loss
    if random.random() < condition['loss'] / 100:
        log_entry(path_id, 0, 0, condition['loss'], 'dropped')
        return 'dropped'

    # Simulasi delay + jitter
    delay = condition['base_delay'] + random.uniform(
        -condition['jitter'],
        condition['jitter']
    )
    delay = max(1, delay)
    time.sleep(delay / 1000)

    try:
        start = time.time()
        response = requests.post(
            f'http://localhost:{port}/path{path_id}',
            json={
                'temperature': round(random.uniform(25, 35), 1),
                'humidity': round(random.uniform(60, 90), 1)
            },
            timeout=5
        )
        rtt = round((time.time() - start) * 1000, 2)
        throughput = 312  # bytes/s konsisten dengan sistem
        log_entry(path_id, rtt, throughput, condition['loss'], 'success')
        return 'success'
    except Exception as e:
        log_entry(path_id, 0, 0, condition['loss'], 'error')
        return 'error'

def run_scenario(scenario):
    print(f"\n[SCENARIO] {scenario['name']} — {scenario['duration_seconds']} detik")
    start_time = time.time()
    count = 0
    transition = scenario.get('transition', None)

    while time.time() - start_time < scenario['duration_seconds']:
        elapsed = time.time() - start_time

        # Hitung kondisi dinamis kalau ada transisi
        cond1 = dict(scenario['path1'])
        cond2 = dict(scenario['path2'])

        if transition:
            t_start = transition['start_at']
            t_end   = transition['end_at']
            t_path  = transition['path']

            if t_start <= elapsed <= t_end:
                progress = (elapsed - t_start) / (t_end - t_start)
                if t_path == 1:
                    cond1['base_delay'] = scenario['path1']['base_delay'] + \
                        progress * (transition['target_delay'] - scenario['path1']['base_delay'])
                    cond1['loss'] = scenario['path1']['loss'] + \
                        progress * (transition['target_loss'] - scenario['path1']['loss'])
                else:
                    cond2['base_delay'] = scenario['path2']['base_delay'] + \
                        progress * (transition['target_delay'] - scenario['path2']['base_delay'])
                    cond2['loss'] = scenario['path2']['loss'] + \
                        progress * (transition['target_loss'] - scenario['path2']['loss'])
            elif elapsed > t_end:
                if t_path == 1:
                    cond1['base_delay'] = transition['target_delay']
                    cond1['loss']       = transition['target_loss']
                else:
                    cond2['base_delay'] = transition['target_delay']
                    cond2['loss']       = transition['target_loss']

        r1 = send_with_condition(1, 5001, cond1)
        r2 = send_with_condition(2, 5002, cond2)
        count += 1

        print(f"  [{elapsed:.1f}s] P1:{r1} | P2:{r2} | total: {count*2} records", end='\r')
        time.sleep(0.5)

    print(f"\n  Selesai: {count*2} records dikumpulkan")

if __name__ == '__main__':
    print("=== Data Generator untuk LSTM Training ===")
    print(f"Total skenario: {len(SCENARIOS)}")
    print(f"Estimasi waktu: {sum(s['duration_seconds'] for s in SCENARIOS) // 60} menit")
    total_est = len(SCENARIOS) * 120 * 2 * 2
    print(f"Estimasi records baru: ~{total_est} records")
    print(f"Data lama akan dipertahankan — records baru ditambahkan\n")

    input("Tekan Enter untuk mulai...")

    for scenario in SCENARIOS:
        run_scenario(scenario)

    print("\n=== Selesai ===")
    print("Cek hasil di: data/logs/path_log.csv")