# simulator/data_generator.py — FIXED VERSION
import requests
import time
import random
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from logger.path_logger import log_entry

# ── Perubahan utama ──────────────────────────────────────────────────────────
# FIX 1: Skenario path2_degraded ditambah dari 1 menjadi 4 variasi
#         supaya distribusi label degradasi lebih seimbang antar path
# FIX 2: throughput tidak lagi di-hardcode 312 — dihitung dinamis
#         berdasarkan delay aktual dan ukuran payload
# FIX 3: Skenario normal dikurangi dari 4 menjadi 2 supaya
#         rasio stable:degraded tidak terlalu timpang
# ────────────────────────────────────────────────────────────────────────────

PAYLOAD_BYTES = 312  # ukuran payload tetap, throughput yang berubah

SCENARIOS = [
    # ── NORMAL (dikurangi dari 4 → 2) ───────────────────────────────────────
    {
        'name': 'normal',
        'duration_seconds': 120,
        'path1': {'base_delay': 20,  'jitter': 5,  'loss': 1.0},
        'path2': {'base_delay': 80,  'jitter': 20, 'loss': 5.0},
    },
    {
        'name': 'normal_stable',
        'duration_seconds': 120,
        'path1': {'base_delay': 18,  'jitter': 3,  'loss': 0.5},
        'path2': {'base_delay': 75,  'jitter': 15, 'loss': 3.0},
    },

    # ── PATH1 DEGRADASI (tetap 3 variasi) ────────────────────────────────────
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
        'name': 'path1_degraded_severe',
        'duration_seconds': 120,
        'path1': {'base_delay': 220, 'jitter': 60, 'loss': 30.0},
        'path2': {'base_delay': 80,  'jitter': 20, 'loss': 5.0},
    },

    # ── PATH2 DEGRADASI (ditambah dari 1 → 4 variasi) ────────────────────────
    # Sebelumnya hanya ada 1 skenario path2_degraded selama 120 detik.
    # Model tidak punya cukup contoh untuk belajar pola degradasi Path2.
    {
        'name': 'path2_degrading_mild',
        'duration_seconds': 120,
        'path1': {'base_delay': 20,  'jitter': 5,  'loss': 1.0},
        'path2': {'base_delay': 120, 'jitter': 35, 'loss': 12.0},
    },
    {
        'name': 'path2_degrading',
        'duration_seconds': 120,
        'path1': {'base_delay': 20,  'jitter': 5,  'loss': 1.0},
        'path2': {'base_delay': 150, 'jitter': 40, 'loss': 18.0},
    },
    {
        'name': 'path2_degraded',
        'duration_seconds': 120,
        'path1': {'base_delay': 20,  'jitter': 5,  'loss': 1.0},
        'path2': {'base_delay': 200, 'jitter': 50, 'loss': 25.0},
    },
    {
        'name': 'path2_degraded_severe',
        'duration_seconds': 120,
        'path1': {'base_delay': 20,  'jitter': 5,  'loss': 1.0},
        'path2': {'base_delay': 250, 'jitter': 60, 'loss': 32.0},
    },

    # ── KEDUA PATH DEGRADASI ──────────────────────────────────────────────────
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

    # ── RECOVERY (dikurangi dari 2 → 1) ──────────────────────────────────────
    {
        'name': 'recovery',
        'duration_seconds': 120,
        'path1': {'base_delay': 25,  'jitter': 8,  'loss': 2.0},
        'path2': {'base_delay': 90,  'jitter': 25, 'loss': 6.0},
    },

    # ── GRADUAL DEGRADATION (penting untuk LSTM belajar tren) ────────────────
    {
        'name': 'gradual_degradation_path1',
        'duration_seconds': 180,
        'path1': {'base_delay': 20,  'jitter': 5,  'loss': 1.0},
        'path2': {'base_delay': 80,  'jitter': 20, 'loss': 5.0},
        'transition': {
            'path': 1, 'start_at': 60, 'end_at': 120,
            'target_delay': 180, 'target_loss': 25.0
        }
    },
    {
        'name': 'gradual_degradation_path2',
        'duration_seconds': 180,
        'path1': {'base_delay': 20,  'jitter': 5,  'loss': 1.0},
        'path2': {'base_delay': 80,  'jitter': 20, 'loss': 5.0},
        'transition': {
            'path': 2, 'start_at': 60, 'end_at': 120,
            'target_delay': 220, 'target_loss': 30.0
        }
    },
    {
        'name': 'gradual_degradation_path2_mild',
        'duration_seconds': 180,
        'path1': {'base_delay': 20,  'jitter': 5,  'loss': 1.0},
        'path2': {'base_delay': 80,  'jitter': 20, 'loss': 5.0},
        'transition': {
            'path': 2, 'start_at': 30, 'end_at': 120,
            'target_delay': 160, 'target_loss': 18.0
        }
    },
    {
        'name': 'path1_high_rtt_low_loss',
        'duration_seconds': 120,
        'path1': {'base_delay': 120, 'jitter': 30, 'loss': 1.0},
        'path2': {'base_delay': 80,  'jitter': 20, 'loss': 5.0},
    },
    {
        'name': 'path1_high_rtt_low_loss_v2',
        'duration_seconds': 120,
        'path1': {'base_delay': 180, 'jitter': 40, 'loss': 2.0},
        'path2': {'base_delay': 80,  'jitter': 20, 'loss': 5.0},
    },
    {
        'name': 'path2_high_rtt_low_loss',
        'duration_seconds': 120,
        'path1': {'base_delay': 20,  'jitter': 5,  'loss': 1.0},
        'path2': {'base_delay': 130, 'jitter': 35, 'loss': 2.0},
    },
    {
        'name': 'path2_high_rtt_low_loss_v2',
        'duration_seconds': 120,
        'path1': {'base_delay': 20,  'jitter': 5,  'loss': 1.0},
        'path2': {'base_delay': 200, 'jitter': 50, 'loss': 3.0},
    },
 
    # ── TAHAP 2: RTT TINGGI, LOSS MULAI MUNCUL ───────────────────────────────
    # Congestion memburuk — loss mulai ada tapi belum parah
    {
        'name': 'path1_high_rtt_mild_loss',
        'duration_seconds': 120,
        'path1': {'base_delay': 130, 'jitter': 35, 'loss': 5.0},
        'path2': {'base_delay': 80,  'jitter': 20, 'loss': 5.0},
    },
    {
        'name': 'path2_high_rtt_mild_loss',
        'duration_seconds': 120,
        'path1': {'base_delay': 20,  'jitter': 5,  'loss': 1.0},
        'path2': {'base_delay': 150, 'jitter': 40, 'loss': 6.0},
    },
    {
        'name': 'path2_high_rtt_mild_loss_v2',
        'duration_seconds': 120,
        'path1': {'base_delay': 20,  'jitter': 5,  'loss': 1.0},
        'path2': {'base_delay': 180, 'jitter': 45, 'loss': 8.0},
    },
 
    # ── GRADUAL: RTT NAIK DULU, LOSS MENYUSUL ────────────────────────────────
    # Ini yang paling penting untuk LSTM — pola tren bertahap
    # yang mencerminkan degradasi jaringan nyata
    {
        'name': 'gradual_rtt_then_loss_path1',
        'duration_seconds': 240,
        'path1': {'base_delay': 20, 'jitter': 5, 'loss': 1.0},
        'path2': {'base_delay': 80, 'jitter': 20, 'loss': 5.0},
        'transition': {
            'path': 1,
            'start_at': 40,
            'end_at': 120,
            # RTT naik dulu ke 180ms tapi loss tetap rendah di 3%
            'target_delay': 180,
            'target_loss': 3.0
        }
    },
    {
        'name': 'gradual_rtt_then_loss_path2',
        'duration_seconds': 240,
        'path1': {'base_delay': 20, 'jitter': 5,  'loss': 1.0},
        'path2': {'base_delay': 80, 'jitter': 20, 'loss': 5.0},
        'transition': {
            'path': 2,
            'start_at': 40,
            'end_at': 120,
            # RTT naik ke 200ms tapi loss hanya 4%
            'target_delay': 200,
            'target_loss': 4.0
        }
    },
 
    # ── JITTER TINGGI SEBAGAI EARLY WARNING ───────────────────────────────────
    # Jitter naik sebelum RTT rata-rata naik — sinyal paling awal
    # dari congestion yang akan datang
    {
        'name': 'path1_high_jitter_prestress',
        'duration_seconds': 120,
        'path1': {'base_delay': 40, 'jitter': 60, 'loss': 1.0},
        'path2': {'base_delay': 80, 'jitter': 20, 'loss': 5.0},
    },
    {
        'name': 'path2_high_jitter_prestress',
        'duration_seconds': 120,
        'path1': {'base_delay': 20,  'jitter': 5,  'loss': 1.0},
        'path2': {'base_delay': 90, 'jitter': 70, 'loss': 3.0},
    },
]


def calculate_throughput(payload_bytes, actual_delay_ms, loss_pct):
    """
    FIX 2: Hitung throughput secara dinamis.
    Throughput turun saat delay tinggi dan loss tinggi.
    throughput = payload / waktu_transmisi * faktor_efektivitas
    """
    delay_s = actual_delay_ms / 1000
    effective_factor = 1.0 - (loss_pct / 100)
    if delay_s <= 0:
        return 0
    throughput = (payload_bytes * effective_factor) / delay_s
    return round(throughput, 2)


def send_with_condition(path_id, port, condition):
    if random.random() < condition['loss'] / 100:
        log_entry(path_id, 0, 0, condition['loss'], 'dropped')
        return 'dropped'

    actual_delay = condition['base_delay'] + random.uniform(
        -condition['jitter'],
        condition['jitter']
    )
    actual_delay = max(1, actual_delay)

    # FIX: mulai hitung RTT SEBELUM sleep supaya delay ikut terhitung
    start = time.time()
    time.sleep(actual_delay / 1000)

    try:
        response = requests.post(
            f'http://localhost:{port}/path{path_id}',
            json={
                'temperature': round(random.uniform(25, 35), 1),
                'humidity':    round(random.uniform(60, 90), 1)
            },
            timeout=5
        )
        rtt = round((time.time() - start) * 1000, 2)
        throughput = calculate_throughput(PAYLOAD_BYTES, actual_delay, condition['loss'])
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
    total_duration = sum(s['duration_seconds'] for s in SCENARIOS)
    print("=== Data Generator untuk LSTM Training — FIXED VERSION ===")
    print(f"Total skenario : {len(SCENARIOS)}")
    print(f"Estimasi waktu : {total_duration // 60} menit")
    print(f"\nDistribusi skenario:")
    print(f"  Normal/stable      : 2 skenario")
    print(f"  Path1 degradasi    : 3 skenario")
    print(f"  Path2 degradasi    : 4 skenario  ← ditambah")
    print(f"  Both degradasi     : 2 skenario")
    print(f"  Recovery           : 1 skenario")
    print(f"  Gradual transition : 3 skenario")
    print(f"\nPERHATIAN: Kalau ingin mulai dari data bersih,")
    print(f"hapus dulu data/logs/path_log2.csv sebelum menjalankan ini.")

    input("\nTekan Enter untuk mulai...")

    for scenario in SCENARIOS:
        run_scenario(scenario)

    print("\n=== Selesai ===")
    print("Cek hasil di: data/logs/path_log2.csv")