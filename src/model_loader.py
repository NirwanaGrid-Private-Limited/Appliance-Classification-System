"""
Model Loader module for loading LightGBM_v4_combined model and verifying feature compatibility.
"""
import os
import json
import joblib
import logging
from typing import Dict, Any, Tuple, List
import pandas as pd

logger = logging.getLogger("live_inference.model_loader")

class ModelLoadError(Exception):
    """Custom exception raised when model or metadata loading fails."""
    pass

class ModelLoader:
    _instance = None
    _model = None
    _metadata = None
    _model_path = None
    _meta_path = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance

    @classmethod
    def load_model(cls, model_dir: str = r"c:\Users\saury\Desktop\nirvana_grid\outputs\models", 
                   model_version: str = "LightGBM_v4_combined", force_reload: bool = False):
        """Loads the joblib model and JSON metadata if not already loaded."""
        if cls._model is not None and not force_reload:
            return cls._model, cls._metadata
            
        cls._model_path = os.path.join(model_dir, f"{model_version}.joblib")
        cls._meta_path = os.path.join(model_dir, f"{model_version}_metadata.json")
        
        if not os.path.exists(cls._model_path):
            raise ModelLoadError(f"Model artifact not found at {cls._model_path}")
        if not os.path.exists(cls._meta_path):
            raise ModelLoadError(f"Model metadata not found at {cls._meta_path}")
            
        try:
            logger.info(f"Loading model from {cls._model_path}...")
            cls._model = joblib.load(cls._model_path)
            with open(cls._meta_path, "r", encoding="utf-8") as f:
                cls._metadata = json.load(f)
                
            expected_features = cls._metadata.get("features", [])
            if not expected_features or len(expected_features) != 15:
                logger.warning(f"Metadata expected features check: found {len(expected_features)} features.")
                
            logger.info(f"Successfully loaded {model_version} (Macro F1={cls._metadata.get('macro_f1')}).")
            return cls._model, cls._metadata
            
        except Exception as e:
            cls._model = None
            cls._metadata = None
            logger.error(f"Failed to load model or metadata: {e}", exc_info=True)
            raise ModelLoadError(f"Error loading model {model_version}: {e}")

    @classmethod
    def get_model(cls):
        """Returns the currently loaded model instance or None."""
        return cls._model

    @classmethod
    def is_loaded(cls) -> bool:
        """Returns True if the model and metadata are currently loaded."""
        return cls._model is not None and cls._metadata is not None

    @classmethod
    def predict(cls, features_df: pd.DataFrame) -> Tuple[str, float, Dict[str, float]]:
        """
        Runs inference using the loaded model.
        Returns:
            - predicted_class: str (e.g. 'AC', 'Geyser', 'LED', 'Refrigerator')
            - confidence: float (max probability from predict_proba)
            - class_probabilities: Dict[str, float]
        """
        if cls._model is None or cls._metadata is None:
            cls.load_model()
            
        try:
            expected_features = cls._metadata.get("features", features_df.columns.tolist())
            missing = [f for f in expected_features if f not in features_df.columns]
            if missing:
                raise ModelLoadError(f"Missing expected features for model prediction: {missing}")
                
            X = features_df[expected_features]
            classes = getattr(cls._model, "classes_", ["AC", "Geyser", "LED", "Refrigerator"])
            
            if hasattr(cls._model, "predict_proba"):
                probs = cls._model.predict_proba(X)[0]
                max_idx = int(np.argmax(probs))
                pred_class = str(classes[max_idx])
                confidence = float(probs[max_idx])
                prob_dict = {str(c): float(p) for c, p in zip(classes, probs)}
            else:
                pred_class = str(cls._model.predict(X)[0])
                confidence = 1.0
                prob_dict = {pred_class: 1.0}
                
            return pred_class, confidence, prob_dict
            
        except Exception as e:
            logger.error(f"Prediction execution error: {e}", exc_info=True)
            raise ModelLoadError(f"Failed during model prediction: {e}")

import numpy as np
