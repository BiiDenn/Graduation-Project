"""
XAI (Explainable AI) Module cho Email Phishing Detection

Sử dụng LIME cho các mô hình Keras và SHAP cho BERT.
"""

from .model_loader import ModelLoader
from .lime_explainer import LIMEExplainer
from .shap_explainer import SHAPExplainer
from .output_handler import OutputHandler
from .xai_pipeline import XAIPipeline

__all__ = [
    "ModelLoader",
    "LIMEExplainer",
    "SHAPExplainer",
    "OutputHandler",
    "XAIPipeline"
]

__version__ = "2.0.0"

