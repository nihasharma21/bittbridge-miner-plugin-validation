"""
LSTM Model Implementation for Bittbridge Miner

This model uses a trained LSTM network to predict USDT/CNY prices 1 hour ahead.
Based on the model developed in outside_model/USDT_CNY_RNN_LSTM.ipynb
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from datetime import datetime, timedelta
import bittensor as bt

try:
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    bt.logging.warning("TensorFlow not available. LSTM model will not work.")

from ..model_interface import PredictionModel


class LSTMModel(PredictionModel):
    """
    LSTM-based model for USDT/CNY price prediction.
    
    Uses the last 12 timesteps (1 hour of 5-minute data) to predict
    the price 1 hour into the future.
    
    Model Architecture:
    - Input: 12 timesteps (1 hour of 5-minute intervals)
    - LSTM layer: 30 units with ReLU activation
    - Dense output layer: 1 unit (price prediction)
    
    Example:
        model = LSTMModel()
        if model.initialize():
            prediction, interval = model.predict("2024-01-15T10:30:00+00:00")
    """
    
    def __init__(self, model_path: str = None, data_path: str = None):
        """
        Initialize the LSTM model.
        
        Args:
            model_path: Path to saved LSTM model file (.h5 or .keras)
                       If None, will look for 'lstm_model.h5' in outside_model directory
            data_path: Path to historical data CSV file
                       If None, will look for 'USDT-CNY_scraper (2).csv' in outside_model directory
        """
        if not TENSORFLOW_AVAILABLE:
            bt.logging.error("TensorFlow is required for LSTMModel but not installed.")
            bt.logging.info("Install with: pip install tensorflow")
        
        # Set default paths relative to this file
        if model_path is None:
            # Go up from example_models/ to miner_model/, then to outside_model/
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, 'outside_model', 'lstm_model.h5')
        
        if data_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_path = os.path.join(base_dir, 'outside_model', 'USDT-CNY_scraper (2).csv')
        
        self.model_path = model_path
        self.data_path = data_path
        self.model = None
        self.historical_data = None
        self.n_steps = 12  # 12 timesteps = 1 hour (5-min intervals)
        self.residual_std = None  # Will be calculated from training residuals
        
    def initialize(self) -> bool:
        """
        Load the trained model and historical data.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        if not TENSORFLOW_AVAILABLE:
            return False
        
        try:
            # Load the trained model
            if not os.path.exists(self.model_path):
                bt.logging.error(f"Model file not found: {self.model_path}")
                bt.logging.info("Please train and save your LSTM model first.")
                bt.logging.info("See WORKFLOW_GUIDE.md for instructions.")
                return False
            
            bt.logging.info(f"Loading LSTM model from {self.model_path}")
            # Load model with compile=False to avoid metrics deserialization issues
            # This is safe since we only need the model for inference, not training
            try:
                self.model = load_model(self.model_path, compile=False)
            except Exception as e:
                # If compile=False doesn't work, try with custom_objects
                bt.logging.debug(f"Loading with compile=False failed: {e}, trying alternative method")
                import tensorflow as tf
                # Try loading with custom objects for metrics
                custom_objects = {
                    'mse': tf.keras.losses.MeanSquaredError(),
                    'mae': tf.keras.metrics.MeanAbsoluteError(),
                }
                self.model = load_model(self.model_path, custom_objects=custom_objects, compile=False)
            bt.logging.success("LSTM model loaded successfully")
            
            # Load historical data
            if not os.path.exists(self.data_path):
                bt.logging.warning(f"Data file not found: {self.data_path}")
                bt.logging.warning("Predictions may fail without historical data.")
                return True  # Model loaded, but data missing
            
            bt.logging.info(f"Loading historical data from {self.data_path}")
            self.historical_data = self._load_historical_data()
            bt.logging.success(f"Loaded {len(self.historical_data)} historical data points")
            
            # Calculate residual standard deviation for confidence intervals
            # This should match the RMSE from your test set evaluation
            # Update this value after training your model
            self.residual_std = 0.002586  # RMSE from LSTM test results in notebook
            
            return True
            
        except Exception as e:
            bt.logging.error(f"Failed to initialize LSTM model: {e}")
            import traceback
            bt.logging.debug(traceback.format_exc())
            return False
    
    def _load_historical_data(self) -> pd.DataFrame:
        """
        Load and prepare historical price data.
        
        Returns:
            DataFrame with datetime index and close_price column
        """
        df = pd.read_csv(self.data_path)
        
        # Detect time and price columns (same logic as notebook)
        time_col = None
        for col in ['timestamp_utc', 'timestamp_local', 'time', 'date', df.columns[0]]:
            if col in df.columns:
                time_col = col
                break
        
        close_col = None
        for col in ['Close', 'close', 'PRICE', 'price']:
            if col in df.columns:
                close_col = col
                break
        
        if close_col is None and len(df.columns) > 1:
            close_col = df.columns[1]
        
        if time_col is None or close_col is None:
            raise ValueError(f"Could not detect time/price columns. Available: {df.columns.tolist()}")
        
        # Prepare dataframe
        df[time_col] = pd.to_datetime(df[time_col])
        time_series_df = df[[time_col, close_col]].copy()
        time_series_df.columns = ['datetime', 'close_price']
        time_series_df = time_series_df.set_index('datetime')
        time_series_df = time_series_df.sort_index()
        
        return time_series_df
    
    def _get_recent_prices(self, timestamp: str, n_steps: int = None) -> Optional[np.ndarray]:
        """
        Get the last n_steps prices before the given timestamp.
        
        Args:
            timestamp: ISO format timestamp string
            n_steps: Number of timesteps to retrieve (default: self.n_steps)
        
        Returns:
            Array of shape (1, n_steps, 1) with recent prices, or None if insufficient data
        """
        if n_steps is None:
            n_steps = self.n_steps
        
        if self.historical_data is None:
            bt.logging.warning("Historical data not loaded")
            return None
        
        try:
            # Parse timestamp
            target_time = pd.to_datetime(timestamp)
            
            # Get data up to (but not including) the target time
            available_data = self.historical_data[self.historical_data.index < target_time]
            
            if len(available_data) < n_steps:
                bt.logging.warning(
                    f"Insufficient historical data: need {n_steps}, have {len(available_data)}"
                )
                return None
            
            # Get the last n_steps prices
            recent_prices = available_data['close_price'].tail(n_steps).values
            
            # Reshape to (1, n_steps, 1) for model input
            return recent_prices.reshape(1, n_steps, 1)
            
        except Exception as e:
            bt.logging.error(f"Error getting recent prices: {e}")
            import traceback
            bt.logging.debug(traceback.format_exc())
            return None
    
    def predict(self, timestamp: str) -> Tuple[Optional[float], Optional[List[float]]]:
        """
        Generate a USDT/CNY price prediction for 1 hour after the given timestamp.
        
        This method:
        1. Retrieves the last 12 timesteps (1 hour) of price data before the timestamp
        2. Feeds this data to the LSTM model
        3. Returns the predicted price and 90% confidence interval
        
        Args:
            timestamp: ISO format timestamp string (e.g., "2024-01-15T10:30:00+00:00")
                     This represents the time from which the prediction should be made.
                     The model predicts the price 1 hour ahead from this timestamp.
        
        Returns:
            Tuple containing:
            - prediction (Optional[float]): The predicted USDT/CNY price 1 hour ahead.
              Returns None if prediction cannot be made.
            - interval (Optional[List[float]]): A list [lower_bound, upper_bound] representing
              the 90% confidence interval for the prediction.
              Returns None if interval cannot be estimated.
        """
        if self.model is None:
            bt.logging.error("Model not initialized. Call initialize() first.")
            return None, None
        
        try:
            # Get recent prices (last 12 timesteps = 1 hour)
            X = self._get_recent_prices(timestamp)
            
            if X is None:
                bt.logging.warning(f"Cannot get recent prices for timestamp {timestamp}")
                return None, None
            
            # Make prediction
            prediction = self.model.predict(X, verbose=0)[0, 0]
            prediction = float(prediction)
            
            # Validate prediction is reasonable
            if not (5.0 <= prediction <= 10.0):  # Sanity check for USDT/CNY
                bt.logging.warning(
                    f"Prediction {prediction:.6f} is outside reasonable range (5-10). "
                    "This may indicate a model issue."
                )
            
            # Calculate 90% confidence interval using residual standard deviation
            # Z-score for 90% confidence interval (two-tailed)
            z_score = 1.64
            
            if self.residual_std is not None:
                margin = z_score * self.residual_std
                lower = prediction - margin
                upper = prediction + margin
                interval = [float(lower), float(upper)]
            else:
                # Fallback: use 1% of prediction as margin
                margin = 0.01 * prediction
                interval = [float(prediction - margin), float(prediction + margin)]
            
            bt.logging.debug(
                f"LSTM prediction for {timestamp}: {prediction:.6f}, "
                f"Interval: [{interval[0]:.6f}, {interval[1]:.6f}]"
            )
            
            return prediction, interval
            
        except Exception as e:
            bt.logging.error(f"Error making prediction: {e}")
            import traceback
            bt.logging.debug(traceback.format_exc())
            return None, None
    
    def cleanup(self) -> None:
        """
        Clean up resources.
        
        Keras models don't need explicit cleanup, but this method is available
        if you need to release resources.
        """
        # Add cleanup logic here if needed
        pass

