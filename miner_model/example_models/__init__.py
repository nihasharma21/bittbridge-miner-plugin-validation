"""
Example Models for Bittbridge Miner

This package contains example implementations of the PredictionModel interface.
These serve as templates for contributors to understand how to implement their own models.
"""

from .simple_model import SimpleAPIModel

# Try to import LSTM model (may fail if TensorFlow not installed)
try:
    from .lstm_model import LSTMModel
    __all__ = ['SimpleAPIModel', 'LSTMModel']
except ImportError:
    __all__ = ['SimpleAPIModel']

