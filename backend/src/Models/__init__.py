"""
Models package for the backend application.

This package contains data model classes for handling ML models and datasets:
- Model: Abstract base class for all model types
- ModelManager: Manages ML model information and metadata
- DatasetManager: Manages dataset information and metadata
"""

from .Model import Model
from .Manager_Models_Model import ModelManager
from .Manager_Dataset_Model import DatasetManager

__all__ = [
    "Model",
    "ModelManager",
    "DatasetManager"
]