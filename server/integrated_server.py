# server/integrated_server.py
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from flask import Flask, request, jsonify
from logger.path_logger import init_logger, log_entry
from model.predictor import PathPredictor
import time
import threading
import random
from datetime import datetime

app = Flask(__name__)
init_logger()
predictor = PathPredictor()

PATH_CONDITIONS = {
    1: {'base_delay_ms': 20,  'jitter_ms': 5,  'packet_loss_pct': 1.0},
    2: {'base_delay_ms': 80,  'jitter_ms': 20, 'packet_loss_pct': 5.0}
}

def simulate_network(path_id):
    cond = PATH_CONDITIONS[path_id]
    if random.random() < cond['packet_loss_pct'] / 100:
        return None, cond['packet_loss_pct'], 'dropped'
    delay = cond['base_delay_ms'] + random.uniform(-cond['jitter_ms'], cond['jitter_ms'])
    delay = max(1, delay)
    time.sleep(delay / 1000)
    return delay, cond['packet_loss_pct'], 'success'

def handle_path(path_id, port_hint=None):
    start   = time.time()
    data    = request.get_json()
    rtt, loss_pct, status = simulate_network(path_id)

    if status == 'dropped':
        throughput = 0
        rtt        = 0
    else:
        rtt        = round(rtt, 2)
        throughput = len(str(data)) * 8

    # Log dan update predictor
    log_entry(path_id, rtt, throughput, loss_pct, status)
    predictor.add_record(path_id, rtt, throughput, loss_pct, status)

    # Ambil prediksi terbaru
    pred1 = predictor.predict(1)
    pred2 = predictor.predict(2)
    rec   = predictor.get_recommendation(pred1, pred2)

    # Bangun response JSON — ini interface ke scheduler teman
    response = {
        'timestamp':      datetime.now().isoformat(),
        'window_seconds': predictor.window,
        'paths': [
            {
                'path_id':     1,
                'description': 'WiFi',
                'current': {
                    'rtt_ms':           rtt if path_id == 1 else None,
                    'throughput_bps':   throughput if path_id == 1 else None,
                    'packet_loss_pct':  loss_pct,
                    'status':          status if path_id == 1 else 'unknown'
                },
                'prediction': pred1 if pred1.get('ready') else {'ready': False}
            },
            {
                'path_id':     2,
                'description': '4G',
                'current': {
                    'rtt_ms':           rtt if path_id == 2 else None,
                    'throughput_bps':   throughput if path_id == 2 else None,
                    'packet_loss_pct':  loss_pct,
                    'status':          status if path_id == 2 else 'unknown'
                },
                'prediction': pred2 if pred2.get('ready') else {'ready': False}
            }
        ],
        'recommendation': rec
    }

    return jsonify(response), 200 if status == 'success' else 503

@app.route('/path1', methods=['POST'])
def path1():
    return handle_path(1)

@app.route('/path2', methods=['POST'])
def path2():
    return handle_path(2)

@app.route('/status', methods=['GET'])
def status():
    """Endpoint untuk scheduler teman — polling prediksi terbaru"""
    pred1 = predictor.predict(1)
    pred2 = predictor.predict(2)
    rec   = predictor.get_recommendation(pred1, pred2)
    return jsonify({
        'timestamp':      datetime.now().isoformat(),
        'paths':          [
            {'path_id': 1, 'prediction': pred1},
            {'path_id': 2, 'prediction': pred2}
        ],
        'recommendation': rec
    })

if __name__ == '__main__':
    print("=== Integrated MP-QUIC + LSTM Server ===")
    print("Path 1 (WiFi) : POST /path1  — port 5000")
    print("Path 2 (4G)   : POST /path2  — port 5000")
    print("Status        : GET  /status — untuk scheduler")
    app.run(host='0.0.0.0', port=5000, debug=False)