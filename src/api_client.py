"""
Production API Client for live PZEM electrical sensor data retrieval.
"""
import os
import requests
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

logger = logging.getLogger("live_inference.api_client")

class APIClientError(Exception):
    """Custom exception raised for API client errors during validation or retrieval."""
    pass

class PZEMAPIClient:
    def __init__(self, base_url: Optional[str] = None, token: Optional[str] = None, timeout_sec: int = 15):
        load_dotenv(override=True)
        self.base_url = base_url or os.getenv("BASE_URL", "https://slhlab.pythonanywhere.com/api/pzem/latest/")
        self.token = token or os.getenv("API_TOKEN")
        self.timeout_sec = int(os.getenv("API_TIMEOUT", timeout_sec))
        
        if not self.base_url.endswith("/"):
            self.base_url += "/"
            
        if not self.token:
            logger.warning("API_TOKEN not found in environment variables or arguments.")
            
    def fetch_latest_reading(self, node_name: str, max_retries: int = 2) -> Dict[str, Any]:
        """
        Fetches the latest reading for `node_name` with retry and validation.
        Returns a validated dictionary of sensor readings.
        Raises APIClientError if all retries fail or response is invalid.
        """
        if not self.token:
            raise APIClientError("Authentication failure: API_TOKEN is not configured.")
            
        url = f"{self.base_url}{node_name}/"
        headers = {"Authorization": f"Bearer {self.token}"}
        
        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                response = requests.get(url, headers=headers, timeout=self.timeout_sec)
                
                # Check authentication
                if response.status_code in (401, 403):
                    raise APIClientError(f"Authentication failure: HTTP {response.status_code} - Invalid Bearer Token.")
                elif response.status_code == 404:
                    raise APIClientError(f"API failure: Endpoint for node {node_name} not found (HTTP 404).")
                elif response.status_code != 200:
                    raise APIClientError(f"API failure: HTTP {response.status_code} - {response.text[:100]}")
                    
                data = response.json()
                self._validate_schema(data, node_name)
                return data
                
            except requests.exceptions.Timeout as e:
                last_exception = e
                logger.warning(f"[Attempt {attempt+1}/{max_retries+1}] Timeout fetching data for {node_name}: {e}")
            except requests.exceptions.ConnectionError as e:
                last_exception = e
                logger.warning(f"[Attempt {attempt+1}/{max_retries+1}] Network error fetching data for {node_name}: {e}")
            except APIClientError:
                # Re-raise authentication or schema validation errors immediately without retrying
                raise
            except Exception as e:
                last_exception = e
                logger.warning(f"[Attempt {attempt+1}/{max_retries+1}] Unexpected error fetching data for {node_name}: {e}")
                
        raise APIClientError(f"Failed to fetch valid data for {node_name} after {max_retries + 1} attempts. Last error: {last_exception}")
        
    def _validate_schema(self, data: Dict[str, Any], expected_node: str):
        """Validates schema structure, data types, and bounds."""
        if not isinstance(data, dict):
            raise APIClientError("Invalid response: Expected JSON payload to be a dictionary.")
            
        if not data.get("found", False):
            raise APIClientError(f"API returned found=False for node {expected_node}.")
            
        required_fields = ["nodeid", "voltage", "current", "power", "pf", "frequency", "relay", "timestamp_ist"]
        missing = [f for f in required_fields if f not in data]
        if missing:
            raise APIClientError(f"Invalid schema: Missing required fields {missing}.")
            
        # Check datatypes & convert
        try:
            voltage = float(data["voltage"])
            current = float(data["current"])
            power = float(data["power"])
            pf = float(data["pf"])
            frequency = float(data["frequency"])
        except (ValueError, TypeError) as e:
            raise APIClientError(f"Incorrect datatypes for numerical sensor fields: {e}")
            
        # Check electrical bounds
        if voltage < 0 or current < 0 or power < 0:
            raise APIClientError(f"Invalid values: Negative electrical measurements found (V={voltage}, I={current}, P={power}).")
        if not (0.0 <= pf <= 1.0):
            raise APIClientError(f"Invalid values: Power factor out of bounds [0, 1] (PF={pf}).")
        if not (40.0 <= frequency <= 65.0):
            raise APIClientError(f"Invalid values: Frequency out of expected grid range (Freq={frequency}Hz).")
