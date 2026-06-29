"""Demo: scan messages and evaluate accuracy + calibration on bundled examples.

Uses the DistilBERT classifier if torch/transformers are installed, otherwise the
torch-free heuristic (the demo prints which one ran).
"""
from scam_spotter import ScamAnalyzer, load_classifier
from scam_spotter.calibration import brier_score, expected_calibration_error
from scam_spotter.data import load_examples

print("=== Scam Spotter Demo ===\n")

classifier = load_classifier()
analyzer = ScamAnalyzer(classifier)
print(f"Classifier: {classifier.source}\n")

samples = [
    "URGENT: Your PayPal account is locked. Verify at http://paypal.secure-verify.xyz within 24h.",
    "Congratulations! You WON a $1000 gift card. Claim now: http://bit.ly/claim-reward",
    "Hi, are we still on for lunch tomorrow at noon?",
]
for text in samples:
    rep = analyzer.analyze(text)
    flags = ", ".join(h.title for h in rep.hits[:3]) or "none"
    print(f"[{rep.risk_level:<8} {rep.risk_score:>3}/100] {text[:52]}…")
    print(f"     model P(scam)={rep.ml_phishing_prob:.0%}  red flags: {flags}")

print("\n--- Accuracy + calibration on bundled examples ---")
examples = load_examples()
probs, labels, correct = [], [], 0
for text, label in examples:
    p = analyzer.analyze(text).ml_phishing_prob
    probs.append(p)
    labels.append(label)
    correct += int((p >= 0.5) == bool(label))

print(f"examples : {len(examples)}")
print(f"accuracy : {correct / len(examples):.1%}")
print(f"Brier    : {brier_score(probs, labels):.4f}")
print(f"ECE      : {expected_calibration_error(probs, labels):.4f}")
print("\nDemo complete.")
