"""
FastAPI Production Service wrapping the Appliance Classification System Inference Pipeline.
Provides REST endpoints for health monitoring, live buffer status, and real-time inference.
"""
import time
import logging
import threading
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv(override=True)

import config
from model_loader import ModelLoader
from rolling_window import RollingWindow
from predictor import Predictor, live_logger
from collector import LiveCollector
from api_client import PZEMAPIClient, APIClientError

# Setup structured logger for FastAPI service
logger = config.setup_logger("fastapi_service", r"c:\Users\saury\Desktop\nirvana_grid\logs\fastapi_service.log")

# Global state tracker for service status & latest prediction
service_state = {
    "collector": None,
    "last_prediction_timestamp": "N/A",
    "last_prediction_result": None,
    "api_connectivity_status": "UNKNOWN"
}

def background_prediction_callback(result: Dict[str, Any]):
    """Callback triggered whenever background collector completes a prediction/poll cycle."""
    service_state["last_prediction_timestamp"] = result.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
    service_state["last_prediction_result"] = result
    if result.get("status") in ["SUCCESS", "WARMING_UP"]:
        service_state["api_connectivity_status"] = "CONNECTED"
    elif result.get("status") == "ERROR":
        service_state["api_connectivity_status"] = f"ERROR: {result.get('error_message')}"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifecycle context manager.
    Loads model once on startup, initializes and starts background polling loop, and shuts down cleanly.
    """
    logger.info("Starting up FastAPI Production Service...")
    live_logger.info("=== FASTAPI SERVICE STARTUP ===")
    
    # 1. Load the model once during startup
    try:
        model, meta = ModelLoader.load_model(force_reload=True)
        logger.info(f"Model successfully loaded on startup: {meta.get('version')}")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}", exc_info=True)
        live_logger.error(f"Startup Model Load Error: {e}")

    # 2. Initialize and start LiveCollector
    collector = LiveCollector(
        node_name=os.getenv("NODE_NAME", "NODE_16"),
        polling_interval_sec=config.POLLING_INTERVAL_SEC,
        window_size_sec=config.WINDOW_SIZE_SEC,
        callback=background_prediction_callback
    )
    service_state["collector"] = collector
    
    polling_thread = threading.Thread(target=collector.start_polling, daemon=True)
    polling_thread.start()
    logger.info("Background collector polling loop initiated.")
    
    yield  # Application is running
    
    # Graceful shutdown
    logger.info("Shutting down FastAPI Production Service gracefully...")
    if service_state["collector"]:
        service_state["collector"].stop()
    live_logger.info("=== FASTAPI SERVICE SHUTDOWN ===")

app = FastAPI(
    title="Appliance Classification System Production API",
    description="Real-time electrical appliance prediction from live aggregated 60-second sensor windows.",
    version="1.0.0",
    lifespan=lifespan
)

# Global Exception Handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception during {request.method} {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "ERROR",
            "error_message": "Internal Server Error during inference service processing.",
            "detail": str(exc),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    )

# Request validation schema for POST /predict
class PredictRequest(BaseModel):
    node_name: Optional[str] = Field(default=os.getenv("NODE_NAME", "NODE_16"), description="Sensor node ID to predict for")
    sample: Optional[Dict[str, Any]] = Field(default=None, description="Optional raw electrical sample dictionary (voltage, current, power, relay) to inject directly into the rolling window")

@app.get("/")
async def root_info():
    """Simple service information endpoint."""
    return {
        "service": "Appliance Classification System Production API",
        "version": "1.0.0",
        "status": "ONLINE",
        "description": "Production inference engine running LightGBM_v4_combined over rolling 60-second windows."
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint returning comprehensive service operational diagnostics.
    """
    collector: Optional[LiveCollector] = service_state.get("collector")
    
    # Model check
    model_loaded = ModelLoader.get_model() is not None
    
    # Buffer status check
    buffer_size = collector.rolling_window.size() if collector else 0
    is_ready = collector.rolling_window.is_ready(min_samples=config.MIN_WINDOW_SAMPLES) if collector else False
    
    # Check API connectivity status via background collector or direct check
    api_status = service_state["api_connectivity_status"]
    if api_status == "UNKNOWN" and collector:
        api_status = "CONNECTING" if collector.is_running() else "STOPPED"
        
    return {
        "service_status": "ONLINE" if model_loaded and (collector and collector.is_running()) else "WARNING",
        "model_loaded_status": model_loaded,
        "api_connectivity_status": api_status,
        "rolling_window_status": "READY" if is_ready else "WARMING_UP",
        "current_buffer_size": buffer_size,
        "last_prediction_timestamp": service_state.get("last_prediction_timestamp", "N/A")
    }

@app.post("/predict")
async def predict_endpoint(req: Optional[PredictRequest] = None):
    """
    Primary prediction endpoint.
    Verifies rolling window readiness against dynamic threshold (ceil(WINDOW_SIZE_SEC / POLLING_INTERVAL_SEC)).
    Returns WARMING_UP if threshold not met, otherwise executes aggregation, feature transformations, and prediction.
    """
    collector: Optional[LiveCollector] = service_state.get("collector")
    if not collector:
        raise HTTPException(status_code=503, detail="Inference collector service is not initialized.")
        
    # If client passed an explicit sample, inject it into the active rolling window
    if req and req.sample:
        try:
            collector.rolling_window.add_sample(req.sample)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid sample payload format: {e}")

    # Verify rolling window readiness
    window_size = collector.rolling_window.size()
    if window_size < config.MIN_WINDOW_SAMPLES:
        return {
            "status": "WARMING_UP",
            "current_samples": window_size,
            "required_samples": config.MIN_WINDOW_SAMPLES,
            "message": f"Rolling window is accumulating samples ({window_size}/{config.MIN_WINDOW_SAMPLES}). Inference suspended until threshold met."
        }

    # Execute prediction from active window
    t0 = time.perf_counter()
    try:
        prediction_result = collector.predictor.predict_from_window(collector.rolling_window)
        
        # If predictor itself returned WARMING_UP or ERROR, handle gracefully
        if prediction_result.get("status") == "WARMING_UP":
            return {
                "status": "WARMING_UP",
                "current_samples": prediction_result.get("window_size", window_size),
                "required_samples": prediction_result.get("min_samples_required", config.MIN_WINDOW_SAMPLES),
                "message": prediction_result.get("message", "Rolling window warming up.")
            }
        elif prediction_result.get("status") == "ERROR":
            raise HTTPException(status_code=500, detail=prediction_result.get("error_message", "Inference failure"))
            
        return {
            "status": "SUCCESS",
            "prediction": prediction_result.get("predicted_appliance"),
            "confidence": prediction_result.get("confidence", 0.0),
            "prediction_timestamp": prediction_result.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S")),
            "inference_latency_ms": prediction_result.get("inference_latency_ms", round((time.perf_counter() - t0) * 1000, 4)),
            "window_size": prediction_result.get("window_size", window_size),
            "aggregated_feature_summary": prediction_result.get("aggregated_summary", {})
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference execution error in POST /predict: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference execution error: {e}")
