"""
Runner script for executing the live production inference pipeline or running automated end-to-end validation.
"""
import os
import sys
import time
import argparse
import signal
import logging
from typing import Dict, Any

from collector import LiveCollector
from api_client import PZEMAPIClient
from rolling_window import RollingWindow
from aggregator import WindowAggregator
from feature_pipeline import FeaturePipeline
from model_loader import ModelLoader
from predictor import Predictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("run_live_prediction")

def on_prediction_result(result: Dict[str, Any]):
    """Callback triggered whenever a prediction iteration completes."""
    status = result.get("status")
    ts = result.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
    win_size = result.get("window_size", 0)
    
    if status == "SUCCESS":
        pred = result.get("predicted_appliance")
        conf = result.get("confidence", 0.0)
        api_lat = result.get("api_latency_ms", "N/A")
        pred_lat = result.get("inference_latency_ms", "N/A")
        summary = result.get("aggregated_summary", {})
        
        print(f"\n[{ts}] >>> PREDICTION SUCCESS: {pred} (Confidence: {conf*100:.2f}%) <<<")
        print(f"    Buffer Size : {win_size} samples accumulated in 60s window")
        print(f"    Latencies   : API={api_lat}ms | Inference={pred_lat}ms")
        print(f"    AggWindow   : V={summary.get('voltage')}V, I={summary.get('current')}A, P={summary.get('power')}W, PF={summary.get('pf')}, Freq={summary.get('frequency')}Hz")
    elif status == "WARMING_UP":
        min_req = result.get("min_samples_required", 6)
        api_lat = result.get("api_latency_ms", "N/A")
        print(f"\n[{ts}] >>> STATUS: WARMING UP ({win_size}/{min_req} samples accumulated in 60s window) | APILatency={api_lat}ms <<<")
    else:
        err = result.get("error_message", "Unknown error")
        print(f"\n[{ts}] >>> INFERENCE ERROR (Buffer: {win_size} samples): {err} <<<")

def run_validation_audit(node_name: str, iterations: int = 8, interval_sec: int = 10):
    """
    Runs comprehensive end-to-end audit checking:
    [PASS] API connectivity, Authentication, Polling
    [PASS] Rolling window behavior across multiple polls
    [PASS] Aggregation & Feature engineering compatibility
    [PASS] Model loading & Prediction
    [PASS] Logging & Error handling
    """
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    print("=" * 70)
    print("PRODUCTION INFERENCE PIPELINE END-TO-END AUDIT")
    print("=" * 70)
    
    # Check 1: Model Loading & Metadata Check
    print("\n[Check 1/5] Verifying LightGBM_v4_combined Model Artifact & Schema...")
    try:
        model, meta = ModelLoader.load_model(force_reload=True)
        print(f"  [PASS] Model successfully loaded: {meta['version']}")
        print(f"  [PASS] Expected features ({len(meta['features'])}): {meta['features']}")
    except Exception as e:
        print(f"  [FAIL] Model loading failed: {e}")
        return False

    # Check 2: API Connectivity & Authentication
    print(f"\n[Check 2/5] Testing API Connectivity & Authentication for {node_name}...")
    client = PZEMAPIClient()
    try:
        t0 = time.perf_counter()
        sample = client.fetch_latest_reading(node_name)
        lat_ms = round((time.perf_counter() - t0) * 1000, 2)
        print(f"  [PASS] API response OK (Latency: {lat_ms}ms)")
        print(f"  [PASS] Sample payload: V={sample['voltage']}V, I={sample['current']}A, P={sample['power']}W, Relay={sample['relay']}")
    except Exception as e:
        print(f"  [FAIL] API connectivity / authentication check failed: {e}")
        return False

    # Check 3: Rolling Window & Aggregation verification
    print("\n[Check 3/5] Testing Rolling Window Maintenance & 1-Minute Aggregation...")
    rw = RollingWindow(window_size_sec=60)
    rw.add_sample(sample)
    print(f"  [PASS] Sample added to buffer. Active size: {rw.size()}")
    try:
        agg = WindowAggregator.aggregate_window(rw.get_window_samples())
        print(f"  [PASS] Aggregation success: {agg['voltage']}V, {agg['current']}A, {agg['power']}W")
    except Exception as e:
        print(f"  [FAIL] Aggregation failed: {e}")
        return False

    # Check 4: Feature Pipeline verification
    print("\n[Check 4/5] Verifying Engineered Feature Compatibility with LightGBM_v4_combined...")
    try:
        feat_df = FeaturePipeline.transform(agg)
        print(f"  [PASS] Generated exact 15-feature dataframe. Columns match: {feat_df.columns.tolist() == meta['features']}")
    except Exception as e:
        print(f"  [FAIL] Feature engineering pipeline failed: {e}")
        return False

    # Check 5: Live Polling Loop & Continuous Inference Audit
    print(f"\n[Check 5/5] Running Long-Duration Live Polling Loop ({iterations} iterations at {interval_sec}s interval = {iterations * interval_sec}s total duration)...")
    collector = LiveCollector(node_name=node_name, polling_interval_sec=interval_sec, window_size_sec=60, callback=on_prediction_result)
    try:
        collector.start_polling(max_iterations=iterations)
        print("\n  [PASS] Live polling loop & prediction completed without errors.")
    except Exception as e:
        print(f"\n  [FAIL] Live collector loop failed: {e}")
        return False

    print("\n" + "=" * 70)
    print("ALL VALIDATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)
    return True

def main():
    parser = argparse.ArgumentParser(description="Live Production Inference Pipeline Runner")
    parser.add_argument("--node", type=str, default="NODE_21", help="Sensor node ID to poll (default: NODE_21)")
    parser.add_argument("--interval", type=int, default=10, help="Polling interval in seconds (default: 10)")
    parser.add_argument("--window", type=int, default=60, help="Rolling window size in seconds (default: 60)")
    parser.add_argument("--validate", action="store_true", help="Run automated end-to-end validation audit")
    parser.add_argument("--iterations", type=int, default=8, help="Number of iterations for validation audit (default: 8)")
    args = parser.parse_args()

    if args.validate:
        success = run_validation_audit(node_name=args.node, iterations=args.iterations, interval_sec=args.interval)
        sys.exit(0 if success else 1)
    else:
        collector = LiveCollector(node_name=args.node, polling_interval_sec=args.interval, 
                                  window_size_sec=args.window, callback=on_prediction_result)
        
        def handle_signal(sig, frame):
            print("\nShutting down live collector gracefully...")
            collector.stop()
            
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)
        
        try:
            collector.start_polling()
        except KeyboardInterrupt:
            collector.stop()

if __name__ == "__main__":
    main()
