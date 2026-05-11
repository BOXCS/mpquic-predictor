# evaluation/compare.py — FIXED VERSION + Matplotlib Visualization
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import time
import random
import csv
from datetime import datetime
from model.predictor import PathPredictor

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (aman untuk server/headless)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

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


# ─────────────────────────────────────────────
# Simulasi & logika utama (tidak berubah)
# ─────────────────────────────────────────────

def simulate_network(base_delay, jitter, loss_pct):
    """Simulasi kondisi jaringan, return (rtt, throughput, status)"""
    if random.random() < loss_pct / 100:
        return None, 0, 'dropped'
    delay = max(1, base_delay + random.uniform(-jitter, jitter))
    time.sleep(delay / 1000)
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

    print(f"    Warming up buffer ({predictor.window} records)...")
    for _ in range(predictor.window + 5):
        for pid, pk in [(1, 'path1'), (2, 'path2')]:
            cond = scenario[pk]
            rtt, throughput, status = simulate_network(
                cond['base_delay'], cond['jitter'], cond['loss']
            )
            predictor.add_record(pid, rtt or 0, throughput, status)
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


# ─────────────────────────────────────────────
# Visualisasi Matplotlib
# ─────────────────────────────────────────────

COLOR_RR   = '#4C72B0'   # biru  — Round Robin
COLOR_LSTM = '#DD8452'   # oranye — LSTM Predictor
COLOR_GRID = '#e8e8e8'

SCENARIO_LABELS = {
    'normal':               'Normal',
    'path1_degraded':       'Path1\nDegraded',
    'path2_degraded':       'Path2\nDegraded',
    'both_degraded':        'Both\nDegraded',
    'path2_high_rtt_low_loss': 'Path2 High\nRTT Low Loss',
    'path1_high_rtt_low_loss': 'Path1 High\nRTT Low Loss',
}


def build_chart_data(summary):
    """Pisahkan summary menjadi dict per-skenario untuk RR dan LSTM."""
    rr   = {s['scenario']: s for s in summary if s['method'] == 'Round Robin'}
    lstm = {s['scenario']: s for s in summary if s['method'] == 'LSTM Predictor'}
    scenarios = [s['name'] for s in TEST_SCENARIOS]
    return scenarios, rr, lstm


def plot_grouped_bar(ax, scenarios, rr_vals, lstm_vals, ylabel, title,
                     fmt='{:.1f}', higher_better=True):
    """Helper: grouped bar chart untuk satu metrik."""
    x     = np.arange(len(scenarios))
    width = 0.35

    bars_rr   = ax.bar(x - width/2, rr_vals,   width, label='Round Robin',
                       color=COLOR_RR,   alpha=0.88, zorder=3)
    bars_lstm = ax.bar(x + width/2, lstm_vals, width, label='LSTM Predictor',
                       color=COLOR_LSTM, alpha=0.88, zorder=3)

    # label nilai di atas bar
    for bar in bars_rr + bars_lstm:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + max(rr_vals + lstm_vals) * 0.01,
                fmt.format(h), ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS.get(s, s) for s in scenarios], fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=8)
    ax.yaxis.grid(True, color=COLOR_GRID, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=8)


def plot_rtt_over_rounds(ax, all_results, scenario_name):
    """Line chart: RTT per ronde untuk satu skenario."""
    for method, color, label in [
        ('round_robin',    COLOR_RR,   'Round Robin'),
        ('lstm_predictor', COLOR_LSTM, 'LSTM Predictor'),
    ]:
        rows = [r for r in all_results
                if r['scenario'] == scenario_name and r['method'] == method]
        rounds = [r['round']  for r in rows]
        rtts   = [r['rtt_ms'] for r in rows]
        ax.plot(rounds, rtts, color=color, alpha=0.75, linewidth=1.2, label=label)

    ax.set_xlabel('Round', fontsize=8)
    ax.set_ylabel('RTT (ms)', fontsize=8)
    ax.set_title(f'RTT per Round — {SCENARIO_LABELS.get(scenario_name, scenario_name)}',
                 fontsize=9, fontweight='bold')
    ax.yaxis.grid(True, color=COLOR_GRID, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=7)


def plot_improvement_heatmap(ax, scenarios, summary):
    """Heatmap: improvement LSTM vs RR (positif = LSTM lebih baik)."""
    rr   = {s['scenario']: s for s in summary if s['method'] == 'Round Robin'}
    lstm = {s['scenario']: s for s in summary if s['method'] == 'LSTM Predictor'}

    metrics      = ['avg_rtt', 'loss_rate']
    metric_labels = ['Avg RTT (ms)\nLower Better', 'Loss Rate (%)\nLower Better']
    data = []
    for m in metrics:
        row = []
        for sc in scenarios:
            # positif = LSTM lebih baik (nilai RR lebih tinggi)
            diff = rr[sc][m] - lstm[sc][m]
            row.append(diff)
        data.append(row)

    data = np.array(data)
    im   = ax.imshow(data, cmap='RdYlGn', aspect='auto')

    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels([SCENARIO_LABELS.get(s, s) for s in scenarios], fontsize=7.5)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metric_labels, fontsize=8)
    ax.set_title('Improvement LSTM vs Round Robin\n(Hijau = LSTM lebih baik)',
                 fontsize=9, fontweight='bold')

    for i in range(len(metrics)):
        for j in range(len(scenarios)):
            val  = data[i, j]
            sign = '+' if val >= 0 else ''
            ax.text(j, i, f'{sign}{val:.1f}', ha='center', va='center',
                    fontsize=8.5, fontweight='bold',
                    color='white' if abs(val) > np.max(np.abs(data)) * 0.6 else 'black')

    plt.colorbar(im, ax=ax, shrink=0.8, label='Δ (RR − LSTM)')


def plot_path_distribution(ax, all_results, scenario_name, method):
    """Pie chart: distribusi path yang dipilih."""
    rows  = [r for r in all_results
             if r['scenario'] == scenario_name and r['method'] == method]
    path1 = sum(1 for r in rows if r['path_chosen'] == 1)
    path2 = sum(1 for r in rows if r['path_chosen'] == 2)

    colors = ['#5499C7', '#E59866']
    ax.pie([path1, path2], labels=['Path 1', 'Path 2'], colors=colors,
           autopct='%1.0f%%', startangle=90, textprops={'fontsize': 8})
    label = 'Round Robin' if method == 'round_robin' else 'LSTM Predictor'
    ax.set_title(f'Path Distribution\n{label} — {SCENARIO_LABELS.get(scenario_name, scenario_name)}',
                 fontsize=8.5, fontweight='bold')


def generate_plots(summary, all_results):
    scenarios, rr, lstm = build_chart_data(summary)

    # ── Figure 1: Ringkasan Komparatif (3 metrik) ──────────────────────────
    fig1, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig1.suptitle('Perbandingan Round Robin vs LSTM Predictor — Semua Skenario',
                  fontsize=13, fontweight='bold', y=1.01)

    plot_grouped_bar(
        axes[0], scenarios,
        [rr[s]['avg_rtt']      for s in scenarios],
        [lstm[s]['avg_rtt']    for s in scenarios],
        'Avg RTT (ms)', 'Average RTT',
        fmt='{:.1f}', higher_better=False
    )
    plot_grouped_bar(
        axes[1], scenarios,
        [rr[s]['loss_rate']   for s in scenarios],
        [lstm[s]['loss_rate'] for s in scenarios],
        'Packet Loss (%)', 'Packet Loss Rate',
        fmt='{:.1f}%', higher_better=False
    )
    plot_grouped_bar(
        axes[2], scenarios,
        [rr[s]['success_rate']   for s in scenarios],
        [lstm[s]['success_rate'] for s in scenarios],
        'Success Rate (%)', 'Success Rate',
        fmt='{:.1f}%', higher_better=True
    )

    plt.tight_layout()
    path1 = os.path.join(RESULT_DIR, 'chart_summary_comparison.png')
    fig1.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"  Grafik 1 disimpan: {path1}")

    # ── Figure 2: RTT per Round (line chart) untuk tiap skenario ───────────
    n_sc  = len(scenarios)
    ncols = 3
    nrows = (n_sc + ncols - 1) // ncols
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(15, nrows * 4))
    axes2 = axes2.flatten()

    for idx, sc in enumerate(scenarios):
        plot_rtt_over_rounds(axes2[idx], all_results, sc)

    # sembunyikan subplot kosong
    for idx in range(n_sc, len(axes2)):
        axes2[idx].set_visible(False)

    fig2.suptitle('RTT per Round — Tiap Skenario', fontsize=12,
                  fontweight='bold', y=1.01)
    plt.tight_layout()
    path2 = os.path.join(RESULT_DIR, 'chart_rtt_per_round.png')
    fig2.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"  Grafik 2 disimpan: {path2}")

    # ── Figure 3: Improvement Heatmap ─────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(11, 3.5))
    plot_improvement_heatmap(ax3, scenarios, summary)
    plt.tight_layout()
    path3 = os.path.join(RESULT_DIR, 'chart_improvement_heatmap.png')
    fig3.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f"  Grafik 3 disimpan: {path3}")

    # ── Figure 4: Path Distribution — skenario 'normal' sebagai contoh ────
    fig4, axes4 = plt.subplots(1, 2, figsize=(9, 4))
    plot_path_distribution(axes4[0], all_results, 'normal', 'round_robin')
    plot_path_distribution(axes4[1], all_results, 'normal', 'lstm_predictor')
    fig4.suptitle('Distribusi Pemilihan Path — Skenario Normal',
                  fontsize=11, fontweight='bold')
    plt.tight_layout()
    path4 = os.path.join(RESULT_DIR, 'chart_path_distribution_normal.png')
    fig4.savefig(path4, dpi=150, bbox_inches='tight')
    plt.close(fig4)
    print(f"  Grafik 4 disimpan: {path4}")

    # ── Figure 5: Radar Chart — profil per-metrik per-skenario ────────────
    _plot_radar(scenarios, rr, lstm)
    path5 = os.path.join(RESULT_DIR, 'chart_radar_profile.png')
    print(f"  Grafik 5 disimpan: {path5}")


def _plot_radar(scenarios, rr, lstm):
    """Radar chart: profil perbandingan RR vs LSTM per skenario."""
    categories   = ['Success Rate', 'Low Loss', 'Low RTT']
    N            = len(categories)
    angles       = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles      += angles[:1]  # tutup lingkaran

    ncols = 3
    nrows = (len(scenarios) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 4.5),
                             subplot_kw=dict(polar=True))
    axes = axes.flatten()

    for idx, sc in enumerate(scenarios):
        ax = axes[idx]

        # Normalisasi 0–100 (higher = better) untuk semua metrik
        max_rtt  = max(rr[sc]['avg_rtt'],  lstm[sc]['avg_rtt'],  1)
        max_loss = max(rr[sc]['loss_rate'], lstm[sc]['loss_rate'], 1)

        def rr_vals():
            return [
                rr[sc]['success_rate'],
                100 - rr[sc]['loss_rate'] / max_loss * 100,
                100 - rr[sc]['avg_rtt']  / max_rtt  * 100,
            ]

        def lstm_vals():
            return [
                lstm[sc]['success_rate'],
                100 - lstm[sc]['loss_rate'] / max_loss * 100,
                100 - lstm[sc]['avg_rtt']   / max_rtt  * 100,
            ]

        for vals, color, label in [
            (rr_vals(),   COLOR_RR,   'Round Robin'),
            (lstm_vals(), COLOR_LSTM, 'LSTM Predictor'),
        ]:
            v = vals + vals[:1]
            ax.plot(angles, v, color=color, linewidth=1.8, label=label)
            ax.fill(angles, v, color=color, alpha=0.20)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_yticklabels([])
        ax.set_title(SCENARIO_LABELS.get(sc, sc), fontsize=9,
                     fontweight='bold', pad=12)
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=7)

    for idx in range(len(scenarios), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Radar Profile — Round Robin vs LSTM Predictor',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, 'chart_radar_profile.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

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

    # ── CSV output ──────────────────────────────────────────────────────────
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

    # ── Visualisasi ─────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("Membuat visualisasi grafik...")
    generate_plots(summary, all_results)

    print(f"\n{'='*50}")
    print(f"Hasil lengkap : {result_file}")
    print(f"Ringkasan     : {summary_file}")
    print("\nGrafik yang dihasilkan:")
    print(f"  1. chart_summary_comparison.png  — Grouped bar (RTT, Loss, Success Rate)")
    print(f"  2. chart_rtt_per_round.png        — Line chart RTT tiap ronde per skenario")
    print(f"  3. chart_improvement_heatmap.png  — Heatmap improvement LSTM vs RR")
    print(f"  4. chart_path_distribution_normal.png — Pie chart distribusi path")
    print(f"  5. chart_radar_profile.png        — Radar chart profil per skenario")
    print("\n=== Evaluasi selesai ===")