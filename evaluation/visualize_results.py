import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# Configuration
DB_PATH = 'data/metrics.db'
COMPARE_RESULTS_PATH = 'evaluation/results fixed/comparison_results.csv'
CHARTS_DIR = 'evaluation/charts'

os.makedirs(CHARTS_DIR, exist_ok=True)

# CSS tokens from ui-context.md
COLOR_LSTM = '#2563eb'       # --chart-path-0 (LSTM)
COLOR_RR = '#16a34a'         # --chart-path-1 (Round Robin)
COLOR_DEGRADED = '#dc2626'   # --chart-degraded / --state-error
COLOR_PREDICTION = '#d97706' # --chart-prediction
COLOR_SUCCESS = '#16a34a'    # --state-success
COLOR_BG = '#f9fafb'         # --bg-base
COLOR_GRID = '#e5e7eb'       # --border-default
COLOR_TEXT = '#111827'       # --text-primary

# Common plot styling
def apply_styling(ax, title, xlabel, ylabel):
    ax.set_title(title, color=COLOR_TEXT, fontweight='bold', pad=15)
    ax.set_xlabel(xlabel, color=COLOR_TEXT)
    ax.set_ylabel(ylabel, color=COLOR_TEXT)
    ax.grid(True, linestyle='--', color=COLOR_GRID, alpha=0.7)
    ax.set_facecolor('#ffffff')
    ax.tick_params(colors=COLOR_TEXT)
    for spine in ax.spines.values():
        spine.set_color(COLOR_GRID)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# 1. rtt_comparison.png
def plot_rtt_comparison():
    print("Generating rtt_comparison.png...")
    df = pd.read_csv(COMPARE_RESULTS_PATH)
    
    # We will pick a scenario that shows switching well, e.g. 'path1_degraded'
    scenario = 'path1_degraded'
    df_scen = df[df['scenario'] == scenario]
    
    df_lstm = df_scen[df_scen['method'] == 'lstm_predictor']
    df_rr = df_scen[df_scen['method'] == 'round_robin']
    
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    
    ax.plot(df_lstm['round'], df_lstm['rtt_ms'], label='LSTM-based Switching', color=COLOR_LSTM, linewidth=2)
    ax.plot(df_rr['round'], df_rr['rtt_ms'], label='Round Robin Baseline', color=COLOR_RR, linewidth=2, alpha=0.7)
    
    apply_styling(ax, f'RTT Comparison: LSTM vs Round Robin ({scenario})', 'Round', 'RTT (ms)')
    ax.legend(loc='upper right', frameon=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'rtt_comparison.png'))
    plt.close()

# 2. degradation_prediction.png
def plot_degradation_prediction():
    print("Generating degradation_prediction.png...")
    conn = sqlite3.connect(DB_PATH)
    
    # Get predictions
    df_pred = pd.read_sql('SELECT timestamp, confidence, degradation_detected FROM prediction_results ORDER BY timestamp', conn)
    df_pred['timestamp'] = pd.to_datetime(df_pred['timestamp'])
    
    # Get actual RTT. Since network_metrics records both paths, let's take average RTT or max RTT per timestamp
    df_metrics = pd.read_sql('SELECT timestamp, path_id, rtt_ms FROM network_metrics ORDER BY timestamp', conn)
    df_metrics['timestamp'] = pd.to_datetime(df_metrics['timestamp'])
    
    # For a clearer plot, just pick path 1 RTT as the 'actual RTT' we are monitoring for degradation
    df_actual = df_metrics[df_metrics['path_id'] == 1].copy()
    
    # Merge asof to align timestamps
    df_merged = pd.merge_asof(df_pred, df_actual, on='timestamp', direction='nearest')
    
    # Limit to first 200 data points for clarity
    df_merged = df_merged.head(200)
    
    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)
    
    # Plot Actual RTT
    ax1.plot(df_merged['timestamp'], df_merged['rtt_ms'], color=COLOR_TEXT, label='Actual RTT (Path 1)', linewidth=1.5)
    ax1.set_ylabel('RTT (ms)', color=COLOR_TEXT)
    ax1.axhline(100.0, color='gray', linestyle=':', label='Degradation Threshold (100ms)')
    
    # Secondary axis for Prediction Probability
    ax2 = ax1.twinx()
    ax2.plot(df_merged['timestamp'], df_merged['confidence'], color=COLOR_PREDICTION, label='Predicted Probability', linewidth=2, linestyle='--')
    ax2.set_ylabel('Degradation Probability', color=COLOR_PREDICTION)
    ax2.set_ylim(0, 1.1)
    
    # Highlight correctly predicted (actual > 100 and pred > threshold)
    actual_degraded = df_merged['rtt_ms'] > 100
    pred_degraded = df_merged['degradation_detected'] == 1
    
    correct = actual_degraded & pred_degraded
    missed = actual_degraded & ~pred_degraded
    
    ax1.fill_between(df_merged['timestamp'], 0, df_merged['rtt_ms'], where=correct, color=COLOR_SUCCESS, alpha=0.3, label='Correctly Predicted')
    ax1.fill_between(df_merged['timestamp'], 0, df_merged['rtt_ms'], where=missed, color=COLOR_DEGRADED, alpha=0.3, label='Missed Prediction')
    
    apply_styling(ax1, 'Actual RTT vs Predicted Degradation Probability', 'Time', 'RTT (ms)')
    
    # Fix legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', frameon=True, fontsize='small')
    
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'degradation_prediction.png'))
    plt.close()
    conn.close()

# 3. switching_events.png
def plot_switching_events():
    print("Generating switching_events.png...")
    conn = sqlite3.connect(DB_PATH)
    # Limit to 50 events to avoid clutter and hanging matplotlib
    df_events = pd.read_sql('SELECT timestamp, from_path, to_path, reason FROM switching_events ORDER BY timestamp LIMIT 50', conn)
    df_events['timestamp'] = pd.to_datetime(df_events['timestamp'])
    conn.close()
    
    fig, ax = plt.subplots(figsize=(12, 4), dpi=300)
    
    # Plot timeline dots
    y_values = np.zeros(len(df_events))
    
    # Separate by to_path for color coding
    path1_events = df_events[df_events['to_path'] == 1]
    path2_events = df_events[df_events['to_path'] == 2]
    
    ax.scatter(path1_events['timestamp'], np.zeros(len(path1_events)), color=COLOR_LSTM, s=100, label='Switched to Path 1', zorder=3)
    ax.scatter(path2_events['timestamp'], np.zeros(len(path2_events)), color=COLOR_RR, s=100, label='Switched to Path 2', zorder=3)
    
    # Draw timeline line
    ax.axhline(0, color=COLOR_TEXT, linewidth=1, zorder=1)
    
    # Annotate with confidence
    for i, row in df_events.iterrows():
        reason = row['reason']
        # Extract confidence if present e.g., "LSTM Prediction: 0.973 >= threshold"
        conf_str = reason
        if '>=' in reason:
            conf_str = reason.split('>=')[0].replace('LSTM Prediction:', '').strip()
        
        offset = 0.05 if i % 2 == 0 else -0.05
        va = 'bottom' if offset > 0 else 'top'
        
        ax.annotate(f"To Path {row['to_path']}\nConf: {conf_str}", 
                    (row['timestamp'], 0),
                    xytext=(0, offset * 500), 
                    textcoords='offset points',
                    ha='center', va=va, fontsize=8,
                    arrowprops=dict(arrowstyle='-', color='gray'))

    apply_styling(ax, 'Path Switching Events Timeline', 'Time', '')
    ax.set_yticks([])
    ax.legend(loc='upper right')
    
    # Format x-axis time
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'switching_events.png'))
    plt.close()

# 4. goodput_comparison.png
def plot_goodput_comparison():
    print("Generating goodput_comparison.png...")
    df = pd.read_csv(COMPARE_RESULTS_PATH)
    
    # Calculate goodput: PAYLOAD_BYTES = 312. goodput = (312 * 8) / (rtt_ms / 1000)
    # If dropped (status != 'success'), goodput = 0
    PAYLOAD_BITS = 312 * 8
    
    def calc_goodput(row):
        if row['status'] != 'success' or row['rtt_ms'] == 0:
            return 0
        return PAYLOAD_BITS / (row['rtt_ms'] / 1000.0)
        
    df['goodput_bps'] = df.apply(calc_goodput, axis=1)
    
    # Group by scenario and method
    grouped = df.groupby(['scenario', 'method'])['goodput_bps'].mean().unstack()
    
    # Rename for plotting
    grouped = grouped.rename(columns={'lstm_predictor': 'LSTM-based Switching', 'round_robin': 'Round Robin'})
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    x = np.arange(len(grouped.index))
    width = 0.35
    
    ax.bar(x - width/2, grouped['Round Robin'], width, label='Round Robin', color=COLOR_RR, alpha=0.8)
    ax.bar(x + width/2, grouped['LSTM-based Switching'], width, label='LSTM-based Switching', color=COLOR_LSTM)
    
    apply_styling(ax, 'Average Goodput under Degradation', 'Scenario', 'Goodput (bps)')
    ax.set_xticks(x)
    ax.set_xticklabels(grouped.index, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'goodput_comparison.png'))
    plt.close()

if __name__ == '__main__':
    print("Generating Evaluation Charts...")
    plot_rtt_comparison()
    plot_degradation_prediction()
    plot_switching_events()
    plot_goodput_comparison()
    print(f"All charts saved to {CHARTS_DIR}/")
