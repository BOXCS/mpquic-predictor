import sys
import os

sys.path.insert(0, "/home/zaky/mpquic-ai")

from server.metric_monitor import MetricMonitor

def test_sliding_window():
    monitor = MetricMonitor()
    window_size = monitor.window_size
    print(f"Window size loaded: {window_size}")

    # Push window_size + 5 mock payloads for path 1
    for i in range(window_size + 5):
        payload = {
            "path_id": 1,
            "temperature": 25.0 + i,
            "humidity": 60.0,
            "rtt_ms": 10.0 + i,
            "loss_pct": 0.0
        }
        monitor.process_payload(payload, "127.0.0.1")

    # Push 3 mock payloads for path 2
    for i in range(3):
        payload = {
            "path_id": 2,
            "temperature": 25.0,
            "humidity": 60.0,
            "rtt_ms": 50.0 + i,
            "loss_pct": 0.0
        }
        monitor.process_payload(payload, "127.0.0.1")

    metrics = monitor.get_latest_metrics()
    
    # Assertions
    assert len(metrics[1]) == window_size, f"Expected {window_size} but got {len(metrics[1])}"
    assert len(metrics[2]) == 3, f"Expected 3 but got {len(metrics[2])}"
    
    # Verify sliding window evicted older items (first RTT should be 15.0 for path 1 if window_size is 20)
    expected_first_rtt = 10.0 + 5 # Because 0..24 were pushed, 5..24 remain (length 20)
    assert metrics[1][0]["rtt_ms"] == expected_first_rtt, f"Expected first RTT {expected_first_rtt}, got {metrics[1][0]['rtt_ms']}"
    
    # Verify goodput calculation: payload is small, let's just ensure it's > 0
    assert metrics[1][-1]["goodput_bps"] > 0, "Goodput should be > 0"
    
    print("ALL TESTS PASSED: Sliding window and metrics generation behaves correctly.")

if __name__ == "__main__":
    test_sliding_window()
