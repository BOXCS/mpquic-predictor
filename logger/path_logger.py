# logger/path_logger.py
import csv
import os
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(__file__), '../data/logs')
LOG_FILE = os.path.join(LOG_DIR, 'path_log.csv')

HEADERS = [
    'timestamp', 'path_id', 'rtt_ms',
    'throughput_bps', 'packet_loss_pct', 'status'
]

def init_logger():
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=HEADERS)
            writer.writeheader()

def log_entry(path_id, rtt_ms, throughput_bps, packet_loss_pct, status):
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writerow({
            'timestamp': datetime.now().isoformat(),
            'path_id': path_id,
            'rtt_ms': rtt_ms,
            'throughput_bps': throughput_bps,
            'packet_loss_pct': packet_loss_pct,
            'status': status
        })