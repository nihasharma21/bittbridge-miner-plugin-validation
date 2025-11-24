#!/usr/bin/env python3
"""
Test script for the LSTM model implementation.

Run this to verify your LSTM model is working correctly before deploying.
"""

import sys
import os

# Add the project root to the path
# File is at: miner_model/outside_model/test_lstm_model.py
# Need to go up 2 levels to get to miner_model/, then up 1 more to get to project root
current_dir = os.path.dirname(os.path.abspath(__file__))  # miner_model/outside_model/
parent_dir = os.path.dirname(current_dir)  # miner_model/
project_root = os.path.dirname(parent_dir)  # project root (bittbridge/)
sys.path.insert(0, project_root)

from miner_model.example_models.lstm_model import LSTMModel


def test_lstm_model():
    """Test the LSTM model."""
    print("=" * 60)
    print("Testing LSTM Model Implementation")
    print("=" * 60)
    
    print("\n1. Initializing LSTM model...")
    model = LSTMModel()
    
    # Test initialization
    print("2. Loading model and data...")
    if not model.initialize():
        print("❌ Model initialization failed!")
        print("\nTroubleshooting:")
        print("  - Make sure you've saved your trained model as 'lstm_model.h5'")
        print("  - Check that the data file exists")
        print("  - Verify TensorFlow is installed: pip install tensorflow")
        return False
    
    print("✅ Model initialized successfully")
    
    # Test prediction with a timestamp from your data
    # Adjust this timestamp to match your data range
    test_timestamp = "2025-10-13T20:00:00+00:00"
    
    print(f"\n3. Testing prediction for timestamp: {test_timestamp}")
    prediction, interval = model.predict(test_timestamp)
    
    # Validate outputs
    if prediction is None:
        print("❌ Prediction returned None")
        print("\nPossible reasons:")
        print("  - Insufficient historical data before the timestamp")
        print("  - Data file doesn't contain data up to that timestamp")
        print("  - Check the timestamp is within your data range")
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
        print("   This might be okay depending on market conditions")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    print("\nYour LSTM model is ready to use!")
    print("\nNext steps:")
    print("  1. Update miner_plugin.py to use LSTMModel")
    print("  2. Or create a custom runner script (see WORKFLOW_GUIDE.md)")
    print("  3. Run your miner on testnet first")
    return True


if __name__ == "__main__":
    success = test_lstm_model()
    sys.exit(0 if success else 1)

