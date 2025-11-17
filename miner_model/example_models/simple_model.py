"""
Simple API-Based Model Example

This is a basic example model that demonstrates how to implement the PredictionModel interface.
It fetches the current USDT/CNY price from CoinGecko API and uses it as a prediction.

This model serves as a starting point for contributors. You can:
1. Replace the API call with your own data source
2. Add preprocessing/feature engineering
3. Implement a more sophisticated prediction algorithm
4. Add caching to reduce API calls
5. Implement proper time series forecasting
"""

import os
import requests
from typing import Tuple, Optional, List
import bittensor as bt

from ..model_interface import PredictionModel


class SimpleAPIModel(PredictionModel):
    """
    A simple example model that fetches current price from CoinGecko API.
    
    This model demonstrates the minimum required implementation:
    - Implements the PredictionModel interface
    - Provides a predict() method that returns (prediction, interval)
    - Handles errors gracefully by returning None
    
    To use this model:
    1. Set COINGECKO_API_KEY environment variable
    2. Instantiate: model = SimpleAPIModel()
    3. Call predict: prediction, interval = model.predict(timestamp)
    """
    
    def __init__(self):
        """
        Initialize the SimpleAPIModel.
        
        In a more sophisticated model, you might:
        - Load pre-trained weights
        - Set up feature engineering pipelines
        """
        self.api_key = os.getenv("COINGECKO_API_KEY")
        if self.api_key is None:
            bt.logging.warning(
                "COINGECKO_API_KEY not found in environment variables. "
                "Model will return None predictions."
            )
    
    def predict(self, timestamp: str) -> Tuple[Optional[float], Optional[List[float]]]:
        """
        Generate a USDT/CNY price prediction.
        
        This is the core method that validators call. It must:
        1. Accept a timestamp string
        2. Return a tuple of (prediction, interval)
        3. Handle errors by returning None values
        
        Args:
            timestamp: ISO format timestamp (e.g., "2024-01-15T10:30:00+00:00")
        
        Returns:
            Tuple of (prediction, interval):
            - prediction: float or None (the predicted price)
            - interval: [lower, upper] or None (90% confidence interval)
        """
        # Step 1: Fetch current price from API
        # In a real model, you might:
        # - Fetch historical data
        # - Apply your ML model
        # - Generate forecast
        
        current_price = self._fetch_current_price()
        
        # Step 2: Handle API failure
        if current_price is None:
            bt.logging.warning(f"Failed to fetch price for timestamp {timestamp}")
            return None, None
        
        # Step 3: Use current price as prediction
        # NOTE: This is a simple example. In practice, you should:
        # - Predict future price (1 hour ahead)
        # - Use historical data and patterns
        # - Apply your trained model
        prediction = current_price
        
        # Step 4: Estimate confidence interval
        # This is a naive approach using fixed volatility.
        # In practice, you should:
        # - Calculate historical volatility
        # - Use prediction uncertainty from your model
        interval = self._estimate_interval(prediction)
        
        # Step 5: Log the prediction (optional, for debugging)
        bt.logging.debug(
            f"Prediction for {timestamp}: {prediction}, Interval: {interval}"
        )
        
        return prediction, interval
    
    def _fetch_current_price(self) -> Optional[float]:
        """
        Helper method to fetch current USDT/CNY price from CoinGecko.
        
        This demonstrates how to:
        - Make API calls
        - Handle errors
        - Return None on failure
        
        Returns:
            float: Current USDT/CNY price, or None if fetch fails
        """
        if self.api_key is None:
            return None
        
        try:
            url = (
                f"https://api.coingecko.com/api/v3/simple/price"
                f"?ids=tether&vs_currencies=cny&precision=4"
                f"&x_cg_demo_api_key={self.api_key}"
            )
            response = requests.get(url, timeout=5)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            data = response.json()
            price = data["tether"]["cny"]
            return float(price)
            
        except requests.exceptions.RequestException as e:
            bt.logging.warning(f"API request failed: {e}")
            return None
        except (KeyError, ValueError) as e:
            bt.logging.warning(f"Failed to parse API response: {e}")
            return None
        except Exception as e:
            bt.logging.error(f"Unexpected error fetching price: {e}")
            return None
    
    def _estimate_interval(self, prediction: float) -> List[float]:
        """
        Helper method to estimate 90% confidence interval.
        
        This is a naive approach using fixed volatility assumption.
        In a real model, you should:
        - Calculate historical volatility from data
        - Use prediction uncertainty from your ML model
        
        Args:
            prediction: The point prediction
        
        Returns:
            List[float]: [lower_bound, upper_bound] for 90% confidence
        """
        # Naive approach: assume 1% standard deviation
        std_dev = 0.01
        
        # Z-score for 90% confidence interval (two-tailed)
        z_score = 1.64
        
        # Calculate bounds
        margin = z_score * std_dev * prediction
        lower = prediction - margin
        upper = prediction + margin
        
        return [lower, upper]

