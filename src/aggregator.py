"""
Aggregator module to aggregate rolling window samples into a clean 1-minute feature vector matching training schema.
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any

logger = logging.getLogger("live_inference.aggregator")

class AggregationError(Exception):
    """Custom exception raised when aggregation fails on rolling window samples."""
    pass

class WindowAggregator:
    @staticmethod
    def aggregate_window(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregates a list of sensor reading dictionaries across the rolling window
        into a single feature dictionary compatible with the training preprocessing pipeline.
        """
        if not samples:
            raise AggregationError("Cannot aggregate an empty rolling window.")
            
        df = pd.DataFrame(samples)
        
        # Ensure numerical types
        num_cols = ["voltage", "current", "power", "pf", "frequency"]
        for col in num_cols:
            if col not in df.columns:
                raise AggregationError(f"Missing numerical sensor field '{col}' for aggregation.")
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
        if df[num_cols].isna().any().any():
            raise AggregationError("NaN values encountered when converting sensor samples to numeric.")
            
        # Map relay to numeric (ON=1, OFF=0)
        if "relay" in df.columns:
            relay_mapped = df["relay"].map({"ON": 1, "OFF": 0, 1: 1, 0: 0}).fillna(0)
            relay_agg = int(relay_mapped.mean() >= 0.5)
        else:
            relay_agg = 0
            
        # Aggregations exactly matching training 1-minute window
        agg_dict = {
            "voltage": float(df["voltage"].mean()),
            "current": float(df["current"].mean()),
            "power": float(df["power"].mean()),
            "pf": float(df["pf"].mean()),
            "frequency": float(df["frequency"].mean()),
            "relay": relay_agg,
            "sample_count": len(df)
        }
        
        # Preserve latest timestamp for diurnal feature extraction
        if "timestamp_ist" in df.columns and not df["timestamp_ist"].isna().all():
            latest_ts_str = str(df["timestamp_ist"].iloc[-1])
            agg_dict["timestamp_ist"] = latest_ts_str
            agg_dict["timestamp_ist_dt"] = pd.to_datetime(latest_ts_str, errors="coerce")
        else:
            now_dt = pd.Timestamp.now()
            agg_dict["timestamp_ist"] = now_dt.strftime("%Y-%m-%d %H:%M:%S")
            agg_dict["timestamp_ist_dt"] = now_dt
            
        # Summary statistics for logging & diagnostics
        agg_dict["_stats"] = {
            "voltage_std": float(df["voltage"].std()) if len(df) > 1 else 0.0,
            "current_std": float(df["current"].std()) if len(df) > 1 else 0.0,
            "power_std": float(df["power"].std()) if len(df) > 1 else 0.0,
            "voltage_min": float(df["voltage"].min()),
            "voltage_max": float(df["voltage"].max()),
            "power_min": float(df["power"].min()),
            "power_max": float(df["power"].max())
        }
        
        logger.debug(f"Aggregated {len(df)} samples into 1-minute vector: V={agg_dict['voltage']:.1f}V, I={agg_dict['current']:.2f}A, P={agg_dict['power']:.1f}W")
        return agg_dict
