"""
LSTM Model Example

This is an example of how to use the helper functions for your model.
Notice how simple it is - you only need to load your model and data!
"""

import os
import pandas as pd
from tensorflow.keras.models import load_model

# Import helper functions (they handle the standard prediction pattern)
from .helpers import predict_1hour_ahead, prepare_dataframe

# ============================================
# SECTION 1: Load Your Model
# ============================================
# Get the directory where this file is located
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, '..', 'outside_model', 'lstm_model.h5')

# Load the model
model = load_model(model_path, compile=False)


# ============================================
# SECTION 2: Load Your Data  
# ============================================
# Path to your data file
data_path = os.path.join(current_dir, '..', 'outside_model', 'USDT-CNY_scraper (2).csv')

# Load and prepare data using helper function
df = pd.read_csv(data_path)
data = prepare_dataframe(df)  # Helper handles all the formatting!


# ============================================
# SECTION 3: Predict Function (USING HELPERS!)
# ============================================
# The predict function uses the standard helper function.
# You can customize the parameters if needed.

def predict(timestamp):
    """
    Predict USDT/CNY price 1 hour ahead.
    
    Uses the standard prediction pattern via helper function.
    Customized to use standard error for confidence interval.
    """
    return predict_1hour_ahead(
        model=model,
        data=data,
        timestamp=timestamp,
        n_steps=12,  # 12 timesteps = 1 hour (for 5-minute data)
        interval_method='std',  # Use standard error method
        interval_std=0.002586  # RMSE from test set evaluation
    )
