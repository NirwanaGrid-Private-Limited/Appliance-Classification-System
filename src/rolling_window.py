"""
Rolling Buffer implementation for maintaining a live 60-second in-memory window of sensor readings.
"""
import time
from datetime import datetime, timezone
import logging
from typing import List, Dict, Any
import config

logger = logging.getLogger("live_inference.rolling_window")

class RollingWindow:
    def __init__(self, window_size_sec: int = getattr(config, "WINDOW_SIZE_SEC", 60)):
        self.window_size_sec = window_size_sec
        self._buffer: List[Dict[str, Any]] = []
        
    def add_sample(self, sample: Dict[str, Any]):
        """
        Stores an incoming API response sample in memory.
        Automatically purges samples older than window_size_sec.
        """
        receipt_time = time.time()
        # Make a copy and attach local receipt timestamp for exact window purging
        stored_sample = dict(sample)
        stored_sample["_receipt_timestamp"] = receipt_time
        
        # Deduplicate only rapid accidental double-calls (< 0.5s gap with exact same sensor values)
        if self._buffer:
            last = self._buffer[-1]
            time_diff = receipt_time - last["_receipt_timestamp"]
            if (time_diff < 0.5 and 
                last.get("timestamp_ist") == stored_sample.get("timestamp_ist") and 
                last.get("power") == stored_sample.get("power") and
                last.get("current") == stored_sample.get("current")):
                last["_receipt_timestamp"] = receipt_time
                self._purge_old(receipt_time)
                return
                
        self._buffer.append(stored_sample)
        self._purge_old(receipt_time)
        logger.debug(f"Added sample to buffer. Buffer size: {len(self._buffer)}")
        
    def _purge_old(self, current_time: float):
        """Discards samples older than window_size_sec."""
        cutoff = current_time - self.window_size_sec
        before_count = len(self._buffer)
        self._buffer = [s for s in self._buffer if s["_receipt_timestamp"] >= cutoff]
        purged = before_count - len(self._buffer)
        if purged > 0:
            logger.debug(f"Purged {purged} old samples beyond {self.window_size_sec}s window.")
            
    def get_window_samples(self) -> List[Dict[str, Any]]:
        """
        Collects all samples from the previous 60 seconds (relative to call instant),
        purges older samples, and returns the active sample list.
        """
        now = time.time()
        self._purge_old(now)
        return list(self._buffer)
        
    def clear(self):
        """Clears the buffer completely."""
        self._buffer.clear()
        
    def size(self) -> int:
        """Returns current number of active samples in the window."""
        self._purge_old(time.time())
        return len(self._buffer)
        
    def is_ready(self, min_samples: int = 1) -> bool:
        """Checks if the rolling window contains sufficient samples for aggregation."""
        return self.size() >= min_samples
