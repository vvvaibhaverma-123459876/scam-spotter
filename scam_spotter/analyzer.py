"""The analyzer: fuse the ML classifier with the explainable rule layer.

Why a hybrid? The transformer is accurate but a black box; the rules are
transparent but brittle. Fusing them gives a score that is both *robust* (the
model generalises to phrasings the rules miss) and *explainable* (every alert
comes with named red flags a human can verify). The fusion is deliberately simple
and documented so the score is auditable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .classifier import BaseClassifier, ClassResult, load_classifier
from .rules import RULES, Hit

# Fusion weights: how much the ML probability vs. the rule evidence drives the
# final risk. The model leads; strong explainable evidence can still push it up.
_ML_WEIGHT = 0.65
_RULE_WEIGHT = 0.35
# Rule score that counts as "fully suspicious" when normalising to [0, 1].
_RULE_SATURATION = 60.0


@dataclass
class Report:
    text: str
    risk_score: int                       # 0..100
    risk_level: str                       # Low | Medium | High | Critical
    ml_phishing_prob: float               # raw model probability
    classifier_source: str                # transformer | heuristic
    hits: List[Hit] = field(default_factory=list)
    advice: List[str] = field(default_factory=list)
    label_probs: dict = field(default_factory=dict)

    @property
    def is_likely_scam(self) -> bool:
        return self.risk_score >= 50

    def to_dict(self) -> dict:
        return {
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "ml_phishing_prob": round(self.ml_phishing_prob, 4),
            "classifier": self.classifier_source,
            "signals": [
                {"id": h.rule_id, "category": h.category, "title": h.title,
                 "weight": h.weight, "evidence": h.evidence}
                for h in self.hits
            ],
            "advice": self.advice,
        }


def _level(score: int) -> str:
    if score >= 80:
        return "Critical"
    if score >= 50:
        return "High"
    if score >= 20:
        return "Medium"
    return "Low"


_BASE_ADVICE = [
    "Don't click links or call numbers in the message — reach the company via its official website or app.",
    "Never share passwords, OTPs, card numbers, or gift-card codes.",
    "Legitimate organisations don't pressure you to act within minutes.",
]


def _advice_for(hits: List[Hit], score: int) -> List[str]:
    advice: List[str] = []
    cats = {h.category for h in hits}
    if "Suspicious link" in cats:
        advice.append("Inspect links before clicking: check the real domain and avoid shortened/IP links.")
    if "Money" in cats:
        advice.append("Stop before paying. Verify any payment request through a known, official channel.")
    if "Data theft" in cats:
        advice.append("Do not enter credentials. Open the service yourself instead of using the provided link.")
    if score >= 50:
        advice.extend(_BASE_ADVICE)
    elif not advice:
        advice.append("No strong scam signals, but stay cautious with unexpected messages.")
    # De-duplicate, keep order.
    seen, out = set(), []
    for a in advice:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


class ScamAnalyzer:
    """Combine a classifier and the rule library into an explainable Report."""

    def __init__(self, classifier: Optional[BaseClassifier] = None, rules=None):
        self.classifier = classifier or load_classifier()
        self.rules = rules if rules is not None else RULES

    def analyze(self, text: str) -> Report:
        text = (text or "").strip()
        if not text:
            return Report(text="", risk_score=0, risk_level="Low",
                          ml_phishing_prob=0.0, classifier_source=self.classifier.source,
                          advice=["Enter a message to analyse."])

        result: ClassResult = self.classifier.predict(text)
        hits = [h for h in (r.check(text) for r in self.rules) if h is not None]
        hits.sort(key=lambda h: h.weight, reverse=True)

        rule_norm = min(1.0, sum(h.weight for h in hits) / _RULE_SATURATION)
        fused = _ML_WEIGHT * result.phishing_prob + _RULE_WEIGHT * rule_norm
        score = int(round(100 * fused))
        score = max(0, min(100, score))

        return Report(
            text=text,
            risk_score=score,
            risk_level=_level(score),
            ml_phishing_prob=result.phishing_prob,
            classifier_source=result.source,
            hits=hits,
            advice=_advice_for(hits, score),
            label_probs=result.label_probs,
        )
