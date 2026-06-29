"""Calibration utilities and metrics.

A classifier that says "95% scam" should be wrong ~5% of the time at that
confidence. Measuring and correcting that is the difference between *using* a
model and *engineering* with one. This module provides:

* ``brier_score`` and ``expected_calibration_error`` (ECE) — standard calibration
  metrics.
* ``reliability_curve`` — binned confidence-vs-accuracy data for plotting.
* ``fit_temperature`` — a 1-D search for the temperature that minimises negative
  log-likelihood (temperature scaling), the standard post-hoc calibration method.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple


def brier_score(probs: Sequence[float], labels: Sequence[int]) -> float:
    """Mean squared error between P(positive) and the binary outcome."""
    if not probs:
        return 0.0
    return sum((p - y) ** 2 for p, y in zip(probs, labels)) / len(probs)


def expected_calibration_error(probs: Sequence[float], labels: Sequence[int],
                               n_bins: int = 10) -> float:
    """ECE: average gap between confidence and accuracy across probability bins."""
    if not probs:
        return 0.0
    n = len(probs)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, p in enumerate(probs) if (lo < p <= hi) or (b == 0 and p == 0.0)]
        if not idx:
            continue
        conf = sum(probs[i] for i in idx) / len(idx)
        acc = sum(labels[i] for i in idx) / len(idx)
        ece += (len(idx) / n) * abs(conf - acc)
    return ece


@dataclass
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    avg_confidence: float
    avg_accuracy: float


def reliability_curve(probs: Sequence[float], labels: Sequence[int],
                      n_bins: int = 10) -> List[ReliabilityBin]:
    bins: List[ReliabilityBin] = []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, p in enumerate(probs) if (lo < p <= hi) or (b == 0 and p == 0.0)]
        if not idx:
            continue
        bins.append(ReliabilityBin(
            lower=lo, upper=hi, count=len(idx),
            avg_confidence=sum(probs[i] for i in idx) / len(idx),
            avg_accuracy=sum(labels[i] for i in idx) / len(idx),
        ))
    return bins


def _nll(probs: Sequence[float], labels: Sequence[int]) -> float:
    eps = 1e-7
    total = 0.0
    for p, y in zip(probs, labels):
        p = min(1 - eps, max(eps, p))
        total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return total / len(probs)


def fit_temperature(logit_pairs: Sequence[Tuple[float, float]], labels: Sequence[int],
                    grid: Sequence[float] | None = None) -> float:
    """Find the temperature minimising NLL for a binary problem.

    Args:
        logit_pairs: (logit_negative, logit_positive) per example.
        labels: 0/1 ground truth.
        grid: temperatures to search.
    Returns the best temperature.
    """
    grid = grid or [round(0.5 + 0.1 * i, 2) for i in range(46)]  # 0.5 .. 5.0
    best_t, best_nll = 1.0, float("inf")
    for t in grid:
        probs = []
        for ln, lp in logit_pairs:
            a, b = ln / t, lp / t
            m = max(a, b)
            ea, eb = math.exp(a - m), math.exp(b - m)
            probs.append(eb / (ea + eb))
        score = _nll(probs, labels)
        if score < best_nll:
            best_nll, best_t = score, t
    return best_t
