# server/server.py
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from flask import Flask, request, jsonify
from logger.path_logger import init_logger, log_entry
import time
import threading
import random

app1 = Flask('path1_server')
app2 = Flask('path2_server')
init_logger()

# Konfigurasi kondisi jaringan per jalur
PATH_CONDITIONS = {
    1: {
        'base_delay_ms': 20,
        'jitter_ms': 5,
        'packet_loss_pct': 1.0,
        'description': 'WiFi - stabil'
    },
    2: {
        'base_delay_ms': 80,
        'jitter_ms': 20,
        'packet_loss_pct': 5.0,
        'description': '4G - variable'
    }
}

def simulate_network(path_id):
    """Simulasi kondisi jaringan dan return metrics"""
    cond = PATH_CONDITIONS[path_id]
    
    # Simulasi packet loss
    if random.random() < cond['packet_loss_pct'] / 100:
        return None, cond['packet_loss_pct'], 'dropped'
    
    # Simulasi delay dengan jitter
    delay = cond['base_delay_ms'] + random.uniform(
        -cond['jitter_ms'], 
        cond['jitter_ms']
    )
    delay = max(1, delay)  # minimal 1ms
    time.sleep(delay / 1000)
    
    return delay, cond['packet_loss_pct'], 'success'

@app1.route('/path1', methods=['POST'])
def path1():
    start = time.time()
    data = request.get_json()
    
    rtt, loss_pct, status = simulate_network(1)
    
    if status == 'dropped':
        log_entry(1, 0, 0, loss_pct, 'dropped')
        return jsonify({'path': 1, 'status': 'dropped'}), 503
    
    throughput = len(str(data)) * 8
    log_entry(1, round(rtt, 2), throughput, loss_pct, status)
    return jsonify({'path': 1, 'status': 'success', 'rtt_ms': round(rtt, 2)})

@app2.route('/path2', methods=['POST'])
def path2():
    start = time.time()
    data = request.get_json()
    
    rtt, loss_pct, status = simulate_network(2)
    
    if status == 'dropped':
        log_entry(2, 0, 0, loss_pct, 'dropped')
        return jsonify({'path': 2, 'status': 'dropped'}), 503
    
    throughput = len(str(data)) * 8
    log_entry(2, round(rtt, 2), throughput, loss_pct, status)
    return jsonify({'path': 2, 'status': 'success', 'rtt_ms': round(rtt, 2)})

def run_path1():
    app1.run(host='127.0.0.1', port=5001, debug=False, use_reloader=False)

def run_path2():
    app2.run(host='127.0.0.1', port=5002, debug=False, use_reloader=False)

if __name__ == '__main__':
    # Cleanup tc rules yang tadi
    os.system('sudo tc qdisc del dev lo root 2>/dev/null')
    
    print("Path 1 (WiFi simulasi) : port 5001 — delay ~20ms, loss 1%")
    print("Path 2 (4G simulasi)   : port 5002 — delay ~80ms, loss 5%")
    
    t1 = threading.Thread(target=run_path1)
    t2 = threading.Thread(target=run_path2)
    t1.daemon = True
    t2.daemon = True
    t1.start()
    t2.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nServer stopped.")