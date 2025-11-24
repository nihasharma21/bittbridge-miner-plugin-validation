"""
LSTM Model Example

This is an example of how to convert your notebook model to the simple function format.
Copy this file and modify it with your own model!
"""

import os
import numpy as np
import pandas as pd

# ============================================
# SECTION 1: Load Your Model
# ============================================
# Path to your saved model (relative to this file)
from tensorflow.keras.models import load_model

# Get the directory where this file is located
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, '..', 'outside_model', 'lstm_model.h5')

# Load the model
model = load_model(model_path, compile=False)


# ============================================
# SECTION 2: Load Your Data  
# ============================================
# Path to your data file (relative to this file)
data_path = os.path.join(current_dir, '..', 'outside_model', 'USDT-CNY_scraper (2).csv')

# Load and prepare data (copy from your notebook)
df = pd.read_csv(data_path)

# Detect time and price columns
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
data = df[[time_col, close_col]].copy()
data.columns = ['datetime', 'close_price']
data = data.set_index('datetime')
data = data.sort_index()


# ============================================
# SECTION 3: Write Predict Function
# ============================================
# This is your prediction logic (copy from notebook)

def get_last_n_prices(data, timestamp, n_steps=12):
    """Get the last n_steps prices before the timestamp"""
    target_time = pd.to_datetime(timestamp)
    available_data = data[data.index < target_time]
    
    if len(available_data) < n_steps:
        return None
    
    recent_prices = available_data['close_price'].tail(n_steps).values
    return recent_prices.reshape(1, n_steps, 1)  # Shape: (1, n_steps, 1)


def predict(timestamp):
    """
    Predict USDT/CNY price 1 hour ahead.
    
    Args:
        timestamp: String like "2024-01-15T10:30:00+00:00"
    
    Returns:
        (prediction, interval) where:
        - prediction: float (the predicted price)
        - interval: [lower, upper] (90% confidence interval)
    """
    # Get recent prices (last 12 timesteps = 1 hour)
    X = get_last_n_prices(data, timestamp, n_steps=12)
    
    if X is None:
        return None, None
    
    # Make prediction
    prediction = model.predict(X, verbose=0)[0, 0]
    prediction = float(prediction)
    
    # Calculate 90% confidence interval
    # Using RMSE from test set evaluation (0.002586)
    residual_std = 0.002586
    z_score = 1.64  # For 90% confidence interval
    margin = z_score * residual_std
    interval = [float(prediction - margin), float(prediction + margin)]
    
    return prediction, interval

