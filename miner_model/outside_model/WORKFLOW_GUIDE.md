# Workflow Guide: Integrating Your LSTM Model into Bittbridge Miner

This guide walks you through integrating your LSTM model (from `USDT_CNY_RNN_LSTM.ipynb`) into the Bittbridge miner plugin system.

## 📋 Overview

Your workflow will involve:
1. **Extracting your model code** from the notebook
2. **Saving your trained model** weights
3. **Creating a model class** that implements the `PredictionModel` interface
4. **Integrating with the miner plugin**
5. **Testing your integration**
6. **Running your miner**

---

## Step 1: Save Your Trained LSTM Model

First, you need to save your trained LSTM model so it can be loaded later.

### Save model

Add this cell to your notebook after training the LSTM model:

```python
# Save the trained LSTM model
model.save('lstm_model.h5')  # or use model.save('lstm_model.keras')
print("Model saved successfully!")
```

---

## Step 2: Create Your Model Class

Create a new file `lstm_model.py` in the `miner_model/example_models/` directory (or create your own models directory).

```python
"""
LSTM Model Implementation for Bittbridge Miner

This model uses a trained LSTM network to predict USDT/CNY prices 1 hour ahead.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from datetime import datetime, timedelta
import bittensor as bt

from tensorflow.keras.models import load_model
from ..model_interface import PredictionModel


class LSTMModel(PredictionModel):
    """
    LSTM-based model for USDT/CNY price prediction.
    
    Uses the last 12 timesteps (1 hour of 5-minute data) to predict
    the price 1 hour into the future.
    """
    
    def __init__(self, model_path: str = None, data_path: str = None):
        """
        Initialize the LSTM model.
        
        Args:
            model_path: Path to saved LSTM model file (.h5 or .keras)
            data_path: Path to historical data CSV file
        """
        # Set default paths relative to this file
        if model_path is None:
            # Adjust path based on your directory structure
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
        try:
            # Load the trained model
            if not os.path.exists(self.model_path):
                bt.logging.error(f"Model file not found: {self.model_path}")
                return False
            
            bt.logging.info(f"Loading LSTM model from {self.model_path}")
            self.model = load_model(self.model_path)
            bt.logging.success("LSTM model loaded successfully")
            
            # Load historical data
            if not os.path.exists(self.data_path):
                bt.logging.warning(f"Data file not found: {self.data_path}. Will need to fetch data.")
                return True  # Model loaded, but data missing
            
            bt.logging.info(f"Loading historical data from {self.data_path}")
            self.historical_data = self._load_historical_data()
            bt.logging.success(f"Loaded {len(self.historical_data)} historical data points")
            
            # Calculate residual standard deviation for confidence intervals
            # You can pre-calculate this from your training/test set
            # For now, we'll use a default value based on your notebook results
            self.residual_std = 0.002586  # RMSE from your LSTM test results
            
            return True
            
        except Exception as e:
            bt.logging.error(f"Failed to initialize LSTM model: {e}")
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
            Array of shape (n_steps,) with recent prices, or None if insufficient data
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
            
            return recent_prices.reshape(1, n_steps, 1)  # Shape: (1, n_steps, 1)
            
        except Exception as e:
            bt.logging.error(f"Error getting recent prices: {e}")
            return None
    
    def predict(self, timestamp: str) -> Tuple[Optional[float], Optional[List[float]]]:
        """
        Generate a USDT/CNY price prediction for 1 hour after the given timestamp.
        
        Args:
            timestamp: ISO format timestamp string (e.g., "2024-01-15T10:30:00+00:00")
        
        Returns:
            Tuple of (prediction, interval):
            - prediction: float or None (predicted price 1 hour ahead)
            - interval: [lower, upper] or None (90% confidence interval)
        """
        if self.model is None:
            bt.logging.error("Model not initialized")
            return None, None
        
        try:
            # Get recent prices (last 12 timesteps = 1 hour)
            X = self._get_recent_prices(timestamp)
            
            if X is None:
                return None, None
            
            # Make prediction
            prediction = self.model.predict(X, verbose=0)[0, 0]
            prediction = float(prediction)
            
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
            return None, None
    
    def cleanup(self) -> None:
        """Clean up resources."""
        # Keras models don't need explicit cleanup, but you can add if needed
        pass
```

---

## Step 3: Update Model Registration

### Duplicate `miner_plugin.py` directly and name it `miner_plugin_LSTM.py`

At the top, change the import:
```python
# Replace this line:
from .example_models.simple_model import SimpleAPIModel

# With this:
from .example_models.lstm_model import LSTMModel
```

And at the bottom (in `__main__`), change:
```python
# Replace:
model = SimpleAPIModel()

# With:
model = LSTMModel()
```

---

## Step 4: Test Your Model

Create a test script `test_lstm_model.py`:

```python
"""Test the LSTM model implementation."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miner_model.example_models.lstm_model import LSTMModel

def test_lstm_model():
    """Test the LSTM model."""
    print("Initializing LSTM model...")
    model = LSTMModel()
    
    # Test initialization
    if not model.initialize():
        print("❌ Model initialization failed!")
        return False
    
    print("✅ Model initialized successfully")
    
    # Test prediction
    # Use a timestamp from your data (but before the end)
    test_timestamp = "2025-10-13T20:00:00+00:00"
    
    print(f"\nTesting prediction for timestamp: {test_timestamp}")
    prediction, interval = model.predict(test_timestamp)
    
    # Validate outputs
    if prediction is None:
        print("❌ Prediction returned None")
        return False
    
    print(f"✅ Prediction: {prediction:.6f}")
    
    if interval is None:
        print("⚠️  Interval is None (but prediction succeeded)")
    else:
        print(f"✅ Interval: [{interval[0]:.6f}, {interval[1]:.6f}]")
        
        # Validate interval
        if len(interval) != 2:
            print("❌ Interval should have 2 elements")
            return False
        
        if interval[0] >= interval[1]:
            print("❌ Lower bound should be less than upper bound")
            return False
        
        if not (interval[0] <= prediction <= interval[1]):
            print("❌ Prediction should be within interval")
            return False
    
    # Validate reasonable price range
    if not (6.0 <= prediction <= 8.0):
        print(f"⚠️  Warning: Prediction {prediction:.6f} is outside typical USDT/CNY range (6-8)")
    
    print("\n✅ All tests passed!")
    return True

if __name__ == "__main__":
    success = test_lstm_model()
    sys.exit(0 if success else 1)
```

Run the test:
```bash
cd /Users/dmitrii/Desktop/miner_plugin/bittbridge
python miner_model/outside_model/test_lstm_model.py
```

---

## Step 5: Install Dependencies

Make sure you have all required dependencies. Check `miner_model/requirements.txt` and add TensorFlow if needed:

```bash
pip install tensorflow pandas numpy
```

---

## Step 6: Run Your Miner

### For testing (testnet):
```bash
cd /Users/dmitrii/Desktop/miner_plugin/bittbridge
python run_lstm_miner.py \
  --netuid 420 \
  --subtensor.network test \
  --wallet.name YOUR_MINER_NAME \
  --wallet.hotkey YOUR_MINER_HOTKEY_NAME \
  --logging.debug
```

### For production (mainnet):
```bash
python run_lstm_miner.py \
  --netuid YOUR_NETUID \
  --subtensor.network finney \
  --wallet.name YOUR_MINER_NAME \
  --wallet.hotkey YOUR_MINER_HOTKEY_NAME
```

---

## 📝 Important Notes

### 1. **Data Updates**
Your model needs access to recent historical data. Consider:
- **Option A**: Keep updating your CSV file with new data
- **Option B**: Fetch data from an API in `_get_recent_prices()`
- **Option C**: Use a database to store historical prices

### 2. **Real-time Data**
The current implementation uses static CSV data. For production, you'll want to:
- Fetch real-time prices from an API
- Update your historical data cache periodically
- Handle cases where data is missing

### 3. **Model Retraining**
Periodically retrain your model with new data to maintain accuracy:
- Set up a cron job or scheduled task
- Retrain weekly/monthly with updated data
- Save new model versions

### 4. **Confidence Intervals**
The current implementation uses a fixed residual standard deviation. Consider:
- Calculating intervals dynamically based on recent prediction errors
- Using prediction intervals from your model if available
- Adjusting intervals based on market volatility

### 5. **Performance**
- Model loading happens once at startup (good!)
- Predictions should be fast (< 1 second)
- Consider caching recent predictions if validators query the same timestamp

---

## 🐛 Troubleshooting

### Issue: Model file not found
- **Solution**: Check the path to `lstm_model.h5` is correct
- Make sure you've saved the model after training

### Issue: Insufficient historical data
- **Solution**: Ensure your CSV file has enough data points before the prediction timestamp
- Consider fetching data from an API if CSV is outdated

### Issue: Predictions are always None
- **Solution**: Check logs for error messages
- Verify your data file path is correct
- Ensure timestamps are in the correct format

### Issue: Import errors
- **Solution**: Make sure you're running from the project root
- Install all dependencies: `pip install -r requirements.txt`
- Check Python path includes the project directory

---

## ✅ Checklist

Before deploying your miner:

- [ ] Trained LSTM model saved (`lstm_model.h5`)
- [ ] Model class created (`lstm_model.py`)
- [ ] Model class implements `PredictionModel` interface
- [ ] `predict()` method returns correct format
- [ ] `initialize()` loads model successfully
- [ ] Test script passes all checks
- [ ] Historical data file accessible
- [ ] Dependencies installed
- [ ] Miner runs without errors
- [ ] Predictions are reasonable (6-8 range for USDT/CNY)

---

## 🚀 Next Steps

1. **Improve data handling**: Add API integration for real-time data
2. **Enhance confidence intervals**: Use dynamic intervals based on volatility
3. **Add monitoring**: Log prediction accuracy and model performance
4. **Optimize performance**: Cache predictions, optimize data loading
5. **Deploy**: Run on a server with good uptime

Good luck with your miner! 🎉

