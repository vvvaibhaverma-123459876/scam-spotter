"""scam-spotter: explainable scam & phishing message detection.

A hybrid AI system: a fine-tuned DistilBERT phishing classifier (the ML core)
fused with a transparent rule layer that explains *why* a message is risky, plus
calibration tooling to keep the model's confidence honest. Runs on CPU, no API
key, deployable as a Gradio app anyone can try.
"""
from .analyzer import Report, ScamAnalyzer
from .classifier import HeuristicClassifier, TransformerClassifier, load_classifier

__version__ = "0.1.0"

__all__ = [
    "ScamAnalyzer",
    "Report",
    "load_classifier",
    "TransformerClassifier",
    "HeuristicClassifier",
]
