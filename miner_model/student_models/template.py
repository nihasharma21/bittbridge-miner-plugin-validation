"""
YOUR MODEL NAME HERE

Copy this file and rename it to your_model.py
Fill in SECTION 1 and SECTION 2, then you're done!
The predict function is already provided - it uses the standard 1-hour-ahead pattern.
"""

import os
import pandas as pd

# Import helper functions (they handle the standard prediction pattern)
from .helpers import predict_1hour_ahead, prepare_dataframe

# ============================================
# SECTION 1: Load Your Model
# ============================================
# Copy your model loading code from notebook here
# Example:
#   from tensorflow.keras.models import load_model
#   model = load_model('my_model.h5')

model = None  # ← Replace with your model loading code


# ============================================
# SECTION 2: Load Your Data  
# ============================================
# Copy your data loading code from notebook here
# The prepare_dataframe() helper will handle formatting automatically
# Example:
#   import pandas as pd
#   df = pd.read_csv('my_data.csv')
#   data = prepare_dataframe(df)  # Helper function handles formatting!

data = None  # ← Replace with your data loading code


# ============================================
# SECTION 3: Predict Function (ALREADY DONE!)
# ============================================
# The standard predict function is provided below.
# It uses the helper function that handles the 1-hour-ahead prediction pattern.
# 
# You can customize it if needed:
#   - Change n_steps (default: 12 = 1 hour for 5-min data)
#   - Change interval_method ('fixed' or 'std')
#   - Change interval_std (standard error for 'std' method)

def predict(timestamp):
    """
    Predict USDT/CNY price 1 hour ahead.
    
    This function uses the standard prediction pattern.
    No need to modify unless you want custom behavior!
    
    Args:
        timestamp: String like "2024-01-15T10:30:00+00:00"
    
    Returns:
        (prediction, interval) where:
        - prediction: float (the predicted price)
        - interval: [lower, upper] (90% confidence interval)
    """
    # Standard 1-hour-ahead prediction
    # Uses last 12 timesteps (1 hour for 5-minute data)
    return predict_1hour_ahead(
        model=model,
        data=data,
        timestamp=timestamp,
        n_steps=12,  # ← Change if your data has different time intervals
        interval_method='fixed',  # ← Change to 'std' if you have std_error
        interval_std=None  # ← Set your std_error here if using 'std' method
    )
