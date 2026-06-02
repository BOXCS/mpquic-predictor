import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.db_writer import (
    write_metrics, write_prediction, write_switching_event, write_sensor_data,
    metrics_engine, sensor_engine,
    NetworkMetric, PredictionResult, SwitchingEvent, SensorReading,
    MetricsSessionLocal, SensorSessionLocal
)

def test_db_writer():
    print("Writing mock data to databases...")
    
    # Write mock data
    write_metrics(path_id=1, rtt_ms=45.2, goodput_bps=1500.0, loss_pct=1.5)
    write_prediction(predicted_path=2, confidence=0.85, degradation_detected=True)
    write_switching_event(from_path=1, to_path=2, reason="LSTM Prediction: RTT Spike")
    write_sensor_data(temperature=28.4, humidity=65.2)
    
    print("Data written. Verifying rows...")
    
    # Verify metrics.db
    with MetricsSessionLocal() as session:
        metrics = session.query(NetworkMetric).all()
        assert len(metrics) > 0, "No network metrics found."
        assert metrics[-1].rtt_ms == 45.2, f"Expected 45.2, got {metrics[-1].rtt_ms}"
        print(f"Verified metrics.db -> NetworkMetric: {len(metrics)} rows")
        
        preds = session.query(PredictionResult).all()
        assert len(preds) > 0, "No prediction results found."
        assert preds[-1].confidence == 0.85, f"Expected 0.85, got {preds[-1].confidence}"
        print(f"Verified metrics.db -> PredictionResult: {len(preds)} rows")
        
        events = session.query(SwitchingEvent).all()
        assert len(events) > 0, "No switching events found."
        assert events[-1].to_path == 2, f"Expected to_path=2, got {events[-1].to_path}"
        print(f"Verified metrics.db -> SwitchingEvent: {len(events)} rows")

    # Verify sensor_data.db
    with SensorSessionLocal() as session:
        sensors = session.query(SensorReading).all()
        assert len(sensors) > 0, "No sensor readings found."
        assert sensors[-1].temperature == 28.4, f"Expected 28.4, got {sensors[-1].temperature}"
        print(f"Verified sensor_data.db -> SensorReading: {len(sensors)} rows")
        
    print("ALL TESTS PASSED: Both databases created and rows inserted successfully.")

if __name__ == "__main__":
    test_db_writer()
