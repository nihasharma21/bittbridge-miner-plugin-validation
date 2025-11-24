"""
Helper Functions for Student Models

These functions handle the standard 1-hour-ahead prediction pattern.
Students can use these instead of writing the predict function from scratch.
"""

import pandas as pd
import numpy as np


def get_recent_prices(data, timestamp, n_steps=12):
    """
    Get the last n_steps prices before the given timestamp.
    
    This is the standard function for getting historical prices needed for prediction.
    Works with any DataFrame that has a datetime index and a 'close_price' column.
    
    Args:
        data: DataFrame with datetime index and 'close_price' column
        timestamp: ISO format timestamp string (e.g., "2024-01-15T10:30:00+00:00")
        n_steps: Number of timesteps to retrieve (default: 12 = 1 hour for 5-min data)
    
    Returns:
        Array of shape (1, n_steps, 1) ready for model input, or None if insufficient data
    
    Example:
        X = get_recent_prices(data, "2024-01-15T10:30:00+00:00", n_steps=12)
        if X is not None:
            prediction = model.predict(X)[0, 0]
    """
    target_time = pd.to_datetime(timestamp)
    available_data = data[data.index < target_time]
    
    if len(available_data) < n_steps:
        return None
    
    recent_prices = available_data['close_price'].tail(n_steps).values
    # Reshape for model input: (1, n_steps, 1)
    return recent_prices.reshape(1, n_steps, 1)


def calculate_interval(prediction, method='fixed', std_error=None, percentage=0.01):
    """
    Calculate 90% confidence interval for a prediction.
    
    Standard function for calculating prediction intervals.
    
    Args:
        prediction: The predicted price (float)
        method: 'fixed' (use percentage) or 'std' (use standard error)
        std_error: Standard error/standard deviation (for 'std' method)
        percentage: Percentage of prediction to use as margin (for 'fixed' method, default: 1%)
    
    Returns:
        List [lower_bound, upper_bound] for 90% confidence interval
    
    Example:
        # Using fixed percentage
        interval = calculate_interval(7.25, method='fixed', percentage=0.01)
        # Returns: [7.1775, 7.3225]
        
        # Using standard error
        interval = calculate_interval(7.25, method='std', std_error=0.002586)
        # Returns: [7.2458, 7.2542]
    """
    z_score = 1.64  # Z-score for 90% confidence interval (two-tailed)
    
    if method == 'std' and std_error is not None:
        margin = z_score * std_error
    else:
        # Fixed percentage method
        margin = percentage * prediction
    
    lower = float(prediction - margin)
    upper = float(prediction + margin)
    
    return [lower, upper]


def prepare_dataframe(df, time_col=None, price_col=None):
    """
    Prepare a DataFrame for time series prediction.
    
    Automatically detects time and price columns, then prepares the DataFrame
    with datetime index and 'close_price' column.
    
    Args:
        df: Raw DataFrame from CSV
        time_col: Name of time column (auto-detected if None)
        price_col: Name of price column (auto-detected if None)
    
    Returns:
        DataFrame with datetime index and 'close_price' column
    
    Example:
        df = pd.read_csv('data.csv')
        data = prepare_dataframe(df)
    """
    # Auto-detect time column
    if time_col is None:
        for col in ['timestamp_utc', 'timestamp_local', 'time', 'date', df.columns[0]]:
            if col in df.columns:
                time_col = col
                break
    
    # Auto-detect price column
    if price_col is None:
        for col in ['Close', 'close', 'PRICE', 'price']:
            if col in df.columns:
                price_col = col
                break
        
        if price_col is None and len(df.columns) > 1:
            price_col = df.columns[1]
    
    if time_col is None or price_col is None:
        raise ValueError(f"Could not detect time/price columns. Available: {df.columns.tolist()}")
    
    # Prepare dataframe
    df[time_col] = pd.to_datetime(df[time_col])
    time_series_df = df[[time_col, price_col]].copy()
    time_series_df.columns = ['datetime', 'close_price']
    time_series_df = time_series_df.set_index('datetime')
    time_series_df = time_series_df.sort_index()
    
    return time_series_df


def predict_1hour_ahead(model, data, timestamp, n_steps=12, interval_method='fixed', interval_std=None):
    """
    Standard 1-hour-ahead prediction function.
    
    This is the complete standard prediction function. Students can use this
    directly or customize it for their needs.
    
    Args:
        model: Trained model (must have .predict() method)
        data: DataFrame with datetime index and 'close_price' column
        timestamp: ISO format timestamp string
        n_steps: Number of timesteps to use (default: 12 = 1 hour for 5-min data)
        interval_method: 'fixed' or 'std' for confidence interval calculation
        interval_std: Standard error for 'std' method
    
    Returns:
        Tuple of (prediction, interval) or (None, None) if prediction fails
    
    Example:
        prediction, interval = predict_1hour_ahead(
            model, data, "2024-01-15T10:30:00+00:00",
            n_steps=12,
            interval_method='std',
            interval_std=0.002586
        )
    """
    # Get recent prices
    X = get_recent_prices(data, timestamp, n_steps=n_steps)
    
    if X is None:
        return None, None
    
    # Make prediction
    try:
        prediction = model.predict(X, verbose=0)[0, 0]
        prediction = float(prediction)
    except Exception:
        return None, None
    
    # Calculate interval
    interval = calculate_interval(
        prediction,
        method=interval_method,
        std_error=interval_std
    )
    
    return prediction, interval

