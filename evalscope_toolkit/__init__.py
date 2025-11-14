"""
Evalscope Toolkit - A simplified toolkit for model evaluation using vLLM and Evalscope

This package provides easy-to-use tools for evaluating language models.
"""

__version__ = "1.0.0"

from .config import EvalConfig
from .dataset_manager import DatasetManager
from .vllm_service import VLLMService
from .evaluator import Evaluator

__all__ = [
    "EvalConfig",
    "DatasetManager",
    "VLLMService",
    "Evaluator",
]
