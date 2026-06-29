"""Classification layer.

Two interchangeable classifiers behind one interface:

* ``TransformerClassifier`` — a fine-tuned DistilBERT phishing detector
  (``cybersectony/phishing-email-detection-distilbert_v2.4.1``, 2.9M downloads).
  This is the AI core: a real 67M-parameter transformer running on CPU.
* ``HeuristicClassifier`` — a torch-free fallback that turns the rule score into a
  probability. It keeps the app (and CI) working with zero ML dependencies and
  serves as an honest baseline to compare the transformer against.

The transformer supports **temperature scaling** (`temperature` arg) so its
confidence can be calibrated — see ``calibration.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

MODEL_ID = "cybersectony/phishing-email-detection-distilbert_v2.4.1"

# Label grouping for the 4-class model (validated empirically against the card):
# phishing classes vs. legitimate classes.
PHISHING_LABEL_IDS = {1, 3}
LABEL_NAMES = {
    0: "legitimate_email",
    1: "phishing_url",
    2: "legitimate_url",
    3: "phishing_email",
}


@dataclass
class ClassResult:
    phishing_prob: float                       # P(scam/phishing) in [0, 1]
    label_probs: Dict[str, float] = field(default_factory=dict)
    source: str = "heuristic"                  # which classifier produced this


class BaseClassifier:
    source = "base"

    def predict(self, text: str) -> ClassResult:  # pragma: no cover - interface
        raise NotImplementedError


def _logistic(x: float, k: float = 0.08, x0: float = 45.0) -> float:
    """Map an unbounded rule score to a probability with a smooth S-curve."""
    return 1.0 / (1.0 + math.exp(-k * (x - x0)))


class HeuristicClassifier(BaseClassifier):
    """Rule-score → probability. Torch-free; used as fallback and baseline."""

    source = "heuristic"

    def __init__(self, rules=None):
        from .rules import RULES

        self._rules = rules if rules is not None else RULES

    def predict(self, text: str) -> ClassResult:
        score = sum(r.weight for r in self._rules if r.check(text) is not None)
        p = _logistic(float(score))
        return ClassResult(phishing_prob=p,
                           label_probs={"phishing": p, "legitimate": 1 - p},
                           source=self.source)


class TransformerClassifier(BaseClassifier):
    """DistilBERT phishing detector with optional temperature-scaled calibration.

    Args:
        model_id: HF model id.
        temperature: softmax temperature (>1 softens overconfident logits).
        device: torch device override.
    """

    source = "transformer"

    def __init__(self, model_id: str = MODEL_ID, temperature: float = 1.0,
                 device: Optional[str] = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.model_id = model_id
        self.temperature = max(1e-3, temperature)
        self.device = device or "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.to(self.device).eval()

    def predict(self, text: str) -> ClassResult:
        torch = self._torch
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.model(**enc).logits[0]
        probs = torch.softmax(logits / self.temperature, dim=-1).tolist()
        label_probs = {LABEL_NAMES.get(i, f"label_{i}"): round(p, 4)
                       for i, p in enumerate(probs)}
        phishing_prob = float(sum(probs[i] for i in PHISHING_LABEL_IDS if i < len(probs)))
        return ClassResult(phishing_prob=phishing_prob, label_probs=label_probs,
                           source=self.source)


def load_classifier(prefer_transformer: bool = True, temperature: float = 1.0) -> BaseClassifier:
    """Return the transformer if torch/transformers are importable, else heuristic."""
    if prefer_transformer:
        try:
            return TransformerClassifier(temperature=temperature)
        except Exception:  # torch missing or model unavailable -> graceful fallback
            pass
    return HeuristicClassifier()
