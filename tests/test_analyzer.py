from scam_spotter.analyzer import ScamAnalyzer
from scam_spotter.classifier import ClassResult, HeuristicClassifier


def _analyzer():
    # Force the torch-free heuristic so tests need no model/torch.
    return ScamAnalyzer(HeuristicClassifier())


def test_obvious_scam_scores_high():
    text = ("URGENT: Your PayPal account is locked. Verify your identity and enter your "
            "OTP at http://paypal.secure-verify.xyz within 24 hours or it will be closed.")
    rep = _analyzer().analyze(text)
    assert rep.risk_score >= 50
    assert rep.is_likely_scam
    assert rep.hits  # explainable signals present
    assert rep.advice


def test_benign_message_scores_low():
    rep = _analyzer().analyze("Hi, are we still on for lunch tomorrow at noon?")
    assert rep.risk_score < 20
    assert rep.risk_level == "Low"
    assert not rep.is_likely_scam


def test_empty_input_handled():
    rep = _analyzer().analyze("   ")
    assert rep.risk_score == 0
    assert rep.risk_level == "Low"


def test_report_to_dict_shape():
    rep = _analyzer().analyze("Send a gift card now to claim your prize http://bit.ly/x")
    d = rep.to_dict()
    assert set(d) >= {"risk_score", "risk_level", "ml_phishing_prob", "signals", "advice"}
    assert isinstance(d["signals"], list)


def test_fusion_uses_ml_probability():
    # A stub classifier that always returns high phishing prob should raise the score
    # even on text with no rule hits.
    class HighProb(HeuristicClassifier):
        source = "stub"

        def predict(self, text):
            return ClassResult(phishing_prob=0.95, label_probs={}, source="stub")

    rep = ScamAnalyzer(HighProb()).analyze("totally innocuous text with no signals")
    assert rep.classifier_source == "stub"
    assert rep.risk_score >= 50  # ML weight alone (0.65 * 0.95) crosses High


def test_levels_monotonic_with_score():
    a = _analyzer()
    low = a.analyze("see you at the game tonight")
    high = a.analyze("URGENT confirm your password and OTP now or account suspended "
                     "http://192.168.1.1/login send gift card bitcoin wire transfer")
    assert high.risk_score > low.risk_score
