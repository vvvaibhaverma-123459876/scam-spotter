"""Bundled labeled examples for demos and calibration evaluation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

_PATH = Path(__file__).with_name("examples.jsonl")


def load_examples() -> List[Tuple[str, int]]:
    """Return (text, label) pairs; label 1 == scam/phishing."""
    out: List[Tuple[str, int]] = []
    for line in _PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        out.append((row["text"], int(row["label"])))
    return out
