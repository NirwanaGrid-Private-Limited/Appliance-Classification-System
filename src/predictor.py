"""
Predictor service orchestrating rolling window aggregation, feature engineering, and inference with structured logging.
"""
import time
import logging
import os
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from aggregator import WindowAggregator, AggregationError
from feature_pipeline import FeaturePipeline, FeatureEngineeringError
from model_loader import ModelLoader, ModelLoadError
from rolling_window import RollingWindow
import config

# Setup dedicated live prediction logger
LOGS_DIR = Path("c:/Users/saury/Desktop/nirvana_grid/logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

live_logger = logging.getLogger("live_prediction")
live_logger.setLevel(logging.INFO)
live_logger.propagate = False
if not live_logger.handlers:
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(LOGS_DIR / "live_prediction.log", encoding="utf-8")
    fh.setFormatter(formatter)
    live_logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    live_logger.addHandler(ch)

class Predictor:
    def __init__(self, model_dir: str = r"c:\Users\saury\Desktop\nirvana_grid\outputs\models", 
                 model_version: str = "LightGBM_v4_combined", min_window_samples: Optional[int] = None):
        self.model_dir = model_dir
        self.model_version = model_version
        self.min_window_samples = min_window_samples if min_window_samples is not None else getattr(config, "MIN_WINDOW_SAMPLES", 6)
        try:
            ModelLoader.load_model(model_dir=self.model_dir, model_version=self.model_version)
        except Exception as e:
            live_logger.error(f"Initialization warning: Could not pre-load model {model_version}: {e}")

    def predict_from_window(self, rolling_window: RollingWindow, api_latency_ms: Optional[float] = None) -> Dict[str, Any]:
        """
        Executes full prediction workflow on the current rolling window buffer.
        Enforces configurable readiness check (min_window_samples). Before threshold is reached,
        returns 'WARMING UP'. Once reached, runs full aggregation and inference.
        Never crashes on failure.
        """
        t_start = time.perf_counter()
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 1. Validate window non-empty
        if not rolling_window.is_ready(min_samples=1):
            msg = "Validation failure: Rolling window buffer is empty (0 samples)."
            live_logger.warning(f"[{timestamp_str}] {msg}")
            return self._build_error_response(msg, timestamp_str, api_latency_ms)
            
        samples = rolling_window.get_window_samples()
        window_size = len(samples)
        
        # 2. Check warm-up readiness threshold
        if window_size < self.min_window_samples:
            live_logger.info(f"WARMING UP | Buffer accumulating: {window_size}/{self.min_window_samples} samples in 60s window | APILatency: {api_latency_ms or 'N/A'}ms")
            return {
                "status": "WARMING_UP",
                "predicted_appliance": "WARMING UP",
                "confidence": 0.0,
                "timestamp": timestamp_str,
                "inference_latency_ms": None,
                "api_latency_ms": round(api_latency_ms, 2) if api_latency_ms is not None else None,
                "window_size": window_size,
                "min_samples_required": self.min_window_samples,
                "message": f"Rolling window warming up ({window_size}/{self.min_window_samples} samples accumulated)"
            }
        
        try:
            # 3. Aggregate across rolling window
            agg_data = WindowAggregator.aggregate_window(samples)
            
            # 3. Feature Engineering
            features_df = FeaturePipeline.transform(agg_data)
            
            # 4. Model Prediction
            pred_class, confidence, prob_dict = ModelLoader.predict(features_df)
            
            t_end = time.perf_counter()
            pred_latency_ms = round((t_end - t_start) * 1000.0, 4)
            
            # Log success details exactly as required
            log_msg = (f"PREDICTION SUCCESS | Node/WindowSize: {window_size} samples | "
                       f"APILatency: {api_latency_ms or 'N/A'}ms | PredLatency: {pred_latency_ms}ms | "
                       f"AggFeatures: V={agg_data['voltage']:.1f}V, I={agg_data['current']:.2f}A, P={agg_data['power']:.1f}W, PF={agg_data['pf']:.2f} | "
                       f"Prediction: {pred_class} (Confidence: {confidence*100:.2f}%)")
            live_logger.info(log_msg)
            
            # Prepare clean feature dict for return output
            feat_dict = {col: float(features_df[col].iloc[0]) for col in features_df.columns}
            
            return {
                "status": "SUCCESS",
                "predicted_appliance": pred_class,
                "confidence": round(confidence, 4),
                "timestamp": timestamp_str,
                "inference_latency_ms": pred_latency_ms,
                "api_latency_ms": round(api_latency_ms, 2) if api_latency_ms is not None else None,
                "window_size": window_size,
                "aggregated_summary": {
                    "voltage": round(agg_data["voltage"], 2),
                    "current": round(agg_data["current"], 4),
                    "power": round(agg_data["power"], 2),
                    "pf": round(agg_data["pf"], 4),
                    "frequency": round(agg_data["frequency"], 2),
                    "relay": agg_data["relay"]
                },
                "engineered_features": feat_dict,
                "class_probabilities": {k: round(v, 4) for k, v in prob_dict.items()}
            }
            
        except (AggregationError, FeatureEngineeringError, ModelLoadError) as e:
            msg = f"Inference pipeline error: {e}"
            live_logger.error(f"[{timestamp_str}] {msg}")
            return self._build_error_response(msg, timestamp_str, api_latency_ms, window_size)
        except Exception as e:
            msg = f"Unexpected runtime error during prediction: {e}"
            live_logger.error(f"[{timestamp_str}] {msg}", exc_info=True)
            return self._build_error_response(msg, timestamp_str, api_latency_ms, window_size)
            
    def _build_error_response(self, error_message: str, timestamp_str: str, 
                              api_latency_ms: Optional[float] = None, window_size: int = 0) -> Dict[str, Any]:
        return {
            "status": "ERROR",
            "predicted_appliance": None,
            "confidence": 0.0,
            "timestamp": timestamp_str,
            "inference_latency_ms": None,
            "api_latency_ms": round(api_latency_ms, 2) if api_latency_ms is not None else None,
            "window_size": window_size,
            "error_message": error_message
        }
