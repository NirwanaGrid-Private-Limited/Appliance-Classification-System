"""
Feature engineering pipeline that transforms aggregated sensor data into the exact 15-feature vector required by LightGBM_v4_combined.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List

logger = logging.getLogger("live_inference.feature_pipeline")

class FeatureEngineeringError(Exception):
    """Custom exception raised when feature generation fails."""
    pass

class FeaturePipeline:
    # Exact feature names and order required by LightGBM_v4_combined
    EXPECTED_FEATURES = [
        "voltage",
        "current",
        "power",
        "pf",
        "frequency",
        "relay",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "is_weekend",
        "apparent_power",
        "reactive_power",
        "current_pf",
        "power_per_voltage"
    ]
    
    @classmethod
    def transform(cls, agg_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Takes an aggregated dictionary and generates a 1-row DataFrame containing
        the exact 15 features matching the LightGBM_v4_combined training schema.
        """
        try:
            voltage = float(agg_data.get("voltage", 0.0))
            current = float(agg_data.get("current", 0.0))
            power = float(agg_data.get("power", 0.0))
            pf = float(agg_data.get("pf", 0.0))
            frequency = float(agg_data.get("frequency", 50.0))
            relay = int(agg_data.get("relay", 0))
            
            ts_dt = agg_data.get("timestamp_ist_dt")
            if not isinstance(ts_dt, pd.Timestamp) or pd.isna(ts_dt):
                ts_dt = pd.Timestamp.now()
                
            hour = ts_dt.hour
            day_of_week = ts_dt.dayofweek
            is_weekend = int(day_of_week >= 5)
            
            hour_sin = float(np.sin(2 * np.pi * hour / 24))
            hour_cos = float(np.cos(2 * np.pi * hour / 24))
            dow_sin = float(np.sin(2 * np.pi * day_of_week / 7))
            dow_cos = float(np.cos(2 * np.pi * day_of_week / 7))
            
            apparent_power = float(voltage * current)
            reactive_sq = apparent_power**2 - power**2
            reactive_power = float(np.sqrt(max(reactive_sq, 0.0)))
            current_pf = float(current * pf)
            
            if voltage > 0:
                power_per_voltage = float(power / voltage)
            else:
                power_per_voltage = 0.0
                
            features_dict = {
                "voltage": [voltage],
                "current": [current],
                "power": [power],
                "pf": [pf],
                "frequency": [frequency],
                "relay": [relay],
                "hour_sin": [hour_sin],
                "hour_cos": [hour_cos],
                "dow_sin": [dow_sin],
                "dow_cos": [dow_cos],
                "is_weekend": [is_weekend],
                "apparent_power": [apparent_power],
                "reactive_power": [reactive_power],
                "current_pf": [current_pf],
                "power_per_voltage": [power_per_voltage]
            }
            
            df_features = pd.DataFrame(features_dict)[cls.EXPECTED_FEATURES]
            logger.debug("Successfully generated 15 engineered features compatible with LightGBM_v4_combined.")
            return df_features
            
        except Exception as e:
            logger.error(f"Error during feature generation: {e}", exc_info=True)
            raise FeatureEngineeringError(f"Failed to generate engineered features: {e}")
