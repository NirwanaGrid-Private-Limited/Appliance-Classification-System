"""
Live Data Collector managing the 10-second API polling loop and feeding the rolling window & predictor.
"""
import time
import signal
import threading
import logging
from typing import Optional, Callable, Dict, Any

from api_client import PZEMAPIClient, APIClientError
from rolling_window import RollingWindow
from predictor import Predictor, live_logger

logger = logging.getLogger("live_inference.collector")

class LiveCollector:
    def __init__(self, node_name: str = "NODE_21", polling_interval_sec: int = 10, 
                 window_size_sec: int = 60, callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.node_name = node_name
        self.polling_interval_sec = polling_interval_sec
        self.api_client = PZEMAPIClient()
        self.rolling_window = RollingWindow(window_size_sec=window_size_sec)
        self.predictor = Predictor()
        self.callback = callback
        self._stop_event = threading.Event()
        self._running = False

    def start_polling(self, max_iterations: Optional[int] = None):
        """
        Starts the polling loop, retrieving sensor readings every `polling_interval_sec` seconds.
        If `max_iterations` is provided, loops for that many iterations before stopping.
        Handles temporary API outages, timeouts, and graceful shutdown.
        """
        self._stop_event.clear()
        self._running = True
        iterations = 0
        
        logger.info(f"Starting live collector loop for {self.node_name} (interval={self.polling_interval_sec}s, window={self.rolling_window.window_size_sec}s)...")
        live_logger.info(f"=== LIVE COLLECTOR STARTED: {self.node_name} ===")
        
        while not self._stop_event.is_set():
            t_loop_start = time.perf_counter()
            api_latency_ms = None
            
            try:
                # 1. Fetch sensor sample from live API
                t_api_start = time.perf_counter()
                sample = self.api_client.fetch_latest_reading(self.node_name, max_retries=1)
                t_api_end = time.perf_counter()
                api_latency_ms = (t_api_end - t_api_start) * 1000.0
                
                # 2. Add sample to rolling window
                self.rolling_window.add_sample(sample)
                
                # 3. Execute prediction
                prediction_result = self.predictor.predict_from_window(self.rolling_window, api_latency_ms=api_latency_ms)
                
                # Attach raw sensor sample for external inspection / report generation if needed
                prediction_result["raw_latest_sample"] = sample
                
                if self.callback:
                    self.callback(prediction_result)
                    
            except APIClientError as e:
                logger.warning(f"API Client error on iteration {iterations+1}: {e}")
                live_logger.warning(f"API Client Error: {e}")
                if self.callback:
                    self.callback({
                        "status": "ERROR",
                        "error_message": str(e),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "window_size": self.rolling_window.size()
                    })
            except Exception as e:
                logger.error(f"Unexpected loop error on iteration {iterations+1}: {e}", exc_info=True)
                live_logger.error(f"Unexpected Loop Error: {e}")
                
            iterations += 1
            if max_iterations is not None and iterations >= max_iterations:
                logger.info(f"Reached max iterations ({max_iterations}). Stopping collector.")
                break
                
            # Sleep remainder of polling interval
            elapsed = time.perf_counter() - t_loop_start
            sleep_time = max(0.0, self.polling_interval_sec - elapsed)
            if not self._stop_event.is_set() and sleep_time > 0:
                self._stop_event.wait(sleep_time)
                
        self._running = False
        logger.info(f"Live collector loop stopped for {self.node_name}. Total iterations: {iterations}")
        live_logger.info(f"=== LIVE COLLECTOR STOPPED: {self.node_name} ===")

    def stop(self):
        """Signals the polling loop to stop gracefully."""
        logger.info("Signal received to stop live collector...")
        self._stop_event.set()

    def is_running(self) -> bool:
        return self._running
