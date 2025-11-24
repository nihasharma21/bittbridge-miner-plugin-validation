"""
Bittbridge Miner Model Plugin Package

This package provides a plugin system for integrating predictive models with the Bittbridge subnet.

Quick Start:
    1. Copy student_models/template.py to student_models/your_model.py
    2. Fill in the 3 sections with your code from notebook
    3. Run: python -m miner_model.miner_plugin --netuid 420 --subtensor.network test ...
"""

from .model_interface import PredictionModel
from .miner_plugin import Miner

__all__ = ['PredictionModel', 'Miner']

