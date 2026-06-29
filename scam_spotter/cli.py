"""Command-line interface for scam-spotter."""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional


def cmd_scan(args: argparse.Namespace) -> int:
    from .analyzer import ScamAnalyzer
    from .classifier import HeuristicClassifier, load_classifier

    text = args.text if args.text is not None else sys.stdin.read()
    classifier = HeuristicClassifier() if args.no_ml else load_classifier()
    report = ScamAnalyzer(classifier).analyze(text)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
        return 0

    print(f"\nRisk: {report.risk_score}/100  [{report.risk_level}]  "
          f"(model P(scam)={report.ml_phishing_prob:.2f}, via {report.classifier_source})")
    if report.hits:
        print("\nRed flags:")
        for h in report.hits:
            ev = f"  e.g. {h.evidence[0]!r}" if h.evidence else ""
            print(f"  • [{h.category}] {h.title} (+{h.weight}){ev}")
    print("\nWhat to do:")
    for a in report.advice:
        print(f"  - {a}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Evaluate classifier accuracy AND calibration on the bundled examples."""
    from .analyzer import ScamAnalyzer
    from .calibration import brier_score, expected_calibration_error
    from .classifier import HeuristicClassifier, load_classifier
    from .data import load_examples

    examples = load_examples()
    classifier = HeuristicClassifier() if args.no_ml else load_classifier()
    analyzer = ScamAnalyzer(classifier)

    probs, labels, correct = [], [], 0
    for text, label in examples:
        rep = analyzer.analyze(text)
        p = rep.ml_phishing_prob
        probs.append(p)
        labels.append(label)
        if (p >= 0.5) == bool(label):
            correct += 1

    acc = correct / len(examples)
    print(f"Classifier : {classifier.source}")
    print(f"Examples   : {len(examples)}")
    print(f"Accuracy   : {acc:.1%}")
    print(f"Brier      : {brier_score(probs, labels):.4f}  (lower = better)")
    print(f"ECE        : {expected_calibration_error(probs, labels):.4f}  (lower = better)")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .app import build_demo

    build_demo().launch(server_name=args.host, server_port=args.port, share=args.share)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scam-spotter",
                                description="Explainable scam & phishing message detector.")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("scan", help="Analyse a message.")
    s.add_argument("text", nargs="?", default=None, help="message text (or pipe via stdin)")
    s.add_argument("--json", action="store_true")
    s.add_argument("--no-ml", action="store_true", help="use the torch-free heuristic only")
    s.set_defaults(func=cmd_scan)

    e = sub.add_parser("eval", help="Accuracy + calibration on the bundled examples.")
    e.add_argument("--no-ml", action="store_true")
    e.set_defaults(func=cmd_eval)

    sv = sub.add_parser("serve", help="Launch the Gradio web app.")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=7860)
    sv.add_argument("--share", action="store_true", help="create a public gradio.live URL")
    sv.set_defaults(func=cmd_serve)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
