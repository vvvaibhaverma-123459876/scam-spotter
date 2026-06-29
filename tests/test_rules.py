from scam_spotter.rules import RULES, Rule


def _fired(text):
    return {r.id for r in RULES if r.check(text) is not None}


def test_urgency_and_threat_detected():
    fired = _fired("URGENT: your account will be suspended within 24 hours, act now.")
    assert "urgency" in fired
    assert "threat" in fired


def test_credential_request_detected():
    assert "credentials" in _fired("Please verify your identity and enter your OTP to continue.")


def test_financial_request_detected():
    assert "financial" in _fired("Send a $50 Amazon gift card to release your prize.")


def test_lookalike_domain_detected():
    assert "lookalike" in _fired("Log in at http://paypal.secure-login.com to fix your account.")


def test_real_brand_domain_not_flagged_as_lookalike():
    assert "lookalike" not in _fired("Your receipt is available at https://www.paypal.com/receipt")


def test_ip_url_detected():
    assert "ip_url" in _fired("Claim your gift at http://192.168.0.1/offer now")


def test_shortener_detected():
    assert "shortener" in _fired("Click here http://bit.ly/abc123 to verify")


def test_risky_tld_detected():
    assert "risky_tld" in _fired("Update billing at http://netflix-billing.online/update")


def test_benign_message_fires_nothing():
    assert _fired("Hi, are we still on for lunch tomorrow at noon?") == set()


def test_evidence_is_captured():
    hit = next(r.check("act now or your account will be suspended") for r in RULES
               if r.id == "urgency")
    assert hit.evidence
    assert any("act now" in e.lower() for e in hit.evidence)
