"""
Bittbridge Miner Model Plugin Package

This package provides a plugin system for integrating predictive models with the Bittbridge subnet.
It includes:
- Model interface definition
- Example model implementations
- Miner plugin that integrates models with the network

Quick Start:
    from miner_model.miner_plugin import Miner
    from miner_model.example_models import SimpleAPIModel
    
    model = SimpleAPIModel()
    miner = Miner(model=model)
    miner.run()
"""

from .model_interface import PredictionModel
from .miner_plugin import Miner

__all__ = ['PredictionModel', 'Miner']

