"""
Model Auto-Discovery Utility

Automatically finds and loads student models from the student_models/ folder.
"""

import os
import importlib
import importlib.util
import bittensor as bt
from typing import Optional
from ..model_interface import PredictionModel
from .function_wrapper import FunctionBasedModel


def load_student_model() -> Optional[PredictionModel]:
    """
    Auto-discover and load student model from student_models/ folder.
    
    Looks for .py files (excluding template.py and __init__.py) and loads
    the first one found. Expects a predict() function in the module.
    
    Returns:
        PredictionModel instance, or None if no model found
    
    Example:
        model = load_student_model()
        if model:
            prediction, interval = model.predict("2024-01-15T10:30:00+00:00")
    """
    # Get the student_models directory path
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    student_models_dir = os.path.join(current_dir, 'student_models')
    
    if not os.path.exists(student_models_dir):
        bt.logging.error(f"Student models directory not found: {student_models_dir}")
        return None
    
    # Find Python files (excluding template and __init__)
    model_files = []
    for file in os.listdir(student_models_dir):
        if file.endswith('.py') and file not in ['template.py', '__init__.py']:
            model_files.append(file)
    
    if not model_files:
        bt.logging.error(
            "No student model found in student_models/ folder.\n"
            "Please copy template.py to your_model.py and fill in your code."
        )
        return None
    
    # Load the first model found
    model_file = model_files[0]
    module_name = model_file[:-3]  # Remove .py extension
    
    bt.logging.info(f"Found student model: {model_file}")
    
    try:
        # Import the module
        spec = importlib.util.spec_from_file_location(
            module_name,
            os.path.join(student_models_dir, model_file)
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Check if predict function exists
        if not hasattr(module, 'predict'):
            bt.logging.error(
                f"Model file {model_file} does not have a predict() function.\n"
                "Please make sure you've filled in SECTION 3 of the template."
            )
            return None
        
        # Wrap the predict function
        model = FunctionBasedModel(module.predict)
        bt.logging.success(f"Successfully loaded model from {model_file}")
        
        return model
        
    except Exception as e:
        bt.logging.error(f"Failed to load model from {model_file}: {e}")
        import traceback
        bt.logging.debug(traceback.format_exc())
        return None


def list_available_models() -> list:
    """
    List all available student models.
    
    Returns:
        List of model file names (excluding template.py)
    """
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    student_models_dir = os.path.join(current_dir, 'student_models')
    
    if not os.path.exists(student_models_dir):
        return []
    
    models = []
    for file in os.listdir(student_models_dir):
        if file.endswith('.py') and file not in ['template.py', '__init__.py']:
            models.append(file)
    
    return models

