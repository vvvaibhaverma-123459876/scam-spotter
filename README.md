---
title: Scam Spotter
emoji: 🛡️
colorFrom: red
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 🛡️ Scam Spotter

**Paste a suspicious text, email, or DM — find out if it's likely a scam, and
*why*.** A hybrid AI system that anyone can use: no install, no API key, runs on
CPU.

👉 **Live demo:** run `python app.py` for a local UI, or `scam-spotter serve --share`
for an instant public link. Designed to deploy to a free Hugging Face Space as-is.

## Why this is more than a keyword filter

It fuses **two complementary engines**:

1. **A fine-tuned DistilBERT phishing classifier** — `cybersectony/phishing-email-detection-distilbert_v2.4.1`
   (67M params, 2.9M downloads). This is the ML core: a real transformer that
   generalises to phrasings a keyword list would miss.
2. **A transparent rule engine** — 16 explainable signals (urgency, credential/OTP
   requests, gift-card & crypto demands, look-alike domains, IP/shortened links,
   channel-switching, secrecy…). This is the *why*: every alert cites the exact
   text that triggered it.

The two are fused into a single 0–100 risk score (`0.65·model + 0.35·rules`), so the
result is both **robust** (the model) and **explainable** (the rules) — the kind of
hybrid design that matters when an automated decision needs to be trusted.

Plus **calibration tooling** (`calibration.py`): Brier score, ECE, reliability
curves, and temperature scaling — because a model that says "95% scam" should be
right ~95% of the time, and measuring/fixing that is an engineering job, not a
given.

## Try it

```bash
pip install -e ".[ml,app]"      # transformer + Gradio UI

python app.py                   # local web UI at http://127.0.0.1:7860
scam-spotter serve --share      # public *.gradio.live link anyone can open

scam-spotter scan "URGENT: verify your PayPal OTP at http://paypal.secure.xyz"
scam-spotter eval               # accuracy + calibration on bundled examples
```

The core library has **zero runtime dependencies**; the transformer and Gradio are
optional extras. Without them, the CLI/app fall back to the explainable heuristic
classifier automatically.

## What it looks like

```
[Critical 100/100] URGENT: Your PayPal account is locked. Verify at http://paypal.secure-verify.xyz…
     model P(scam)=100%  red flags: Look-alike brand domain, False urgency, Impersonation

[Low        0/100] Hi, are we still on for lunch tomorrow at noon?
     model P(scam)=0%   red flags: none
```

`scam-spotter eval` on the bundled labelled set (mixed email + SMS scams):
```
classifier : transformer
accuracy   : 81.8%
Brier      : 0.1749
ECE        : 0.1793   ← the model is well-trained on phishing *emails*; SMS-style
                         scams are slightly out-of-distribution, which the
                         calibration tooling is built to measure and correct.
```

## Architecture

```
                 ┌─────────────────────────┐
   message  ───► │   TransformerClassifier  │ ── P(scam) ──┐
                 │   (DistilBERT, CPU)      │              │
                 └─────────────────────────┘              ▼
                 ┌─────────────────────────┐        ┌───────────┐
                 │   Rule engine (16)       │ ─ hits►│ ScamAnalyzer│─► Report
                 │   explainable signals    │        │  (fusion)  │   (score, level,
                 └─────────────────────────┘        └───────────┘    red flags, advice)
   calibration.py: Brier / ECE / reliability / temperature scaling (confidence, audited)
```

Every layer is swappable: `BaseClassifier` lets you drop in a different model or
the torch-free `HeuristicClassifier`; the rule set is plain data.

## Testing

```bash
pip install -e ".[dev]"
pytest -q       # 22 tests, torch-free (heuristic path), run in CI on 3.10–3.12
```

## Deploy as a Hugging Face Space

This repo is Space-ready (the YAML header above + `app.py` + `requirements.txt`).
Create a Gradio Space and push, or:

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli upload <user>/scam-spotter . --repo-type space
```

## Privacy & scope

Messages are analysed in-memory and never stored. This is an educational aid, not
security or legal advice — when in doubt, contact the organisation through its
official website or app.

## License

MIT
