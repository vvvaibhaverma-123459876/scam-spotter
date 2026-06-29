"""The rule library: explainable scam/phishing signals.

Each rule is a self-contained, human-readable detector. A rule fires when its
pattern matches the message; the weight expresses how strongly that signal
indicates a scam. Everything is pure-Python/regex — no model, no API key — so it
runs instantly, offline, and is fully testable.

The goal is *explainability*: every point in the risk score traces back to a named
signal with the exact text that triggered it, so a non-technical user learns what
to look for next time.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Pattern


@dataclass
class Hit:
    """A single fired signal with the evidence that triggered it."""

    rule_id: str
    category: str
    weight: int
    title: str
    explanation: str
    evidence: List[str] = field(default_factory=list)


@dataclass
class Rule:
    id: str
    category: str
    weight: int
    title: str
    explanation: str
    patterns: List[str] = field(default_factory=list)
    custom: Optional[Callable[[str], List[str]]] = None
    _compiled: List[Pattern] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._compiled = [re.compile(p, re.IGNORECASE) for p in self.patterns]

    def check(self, text: str) -> Optional[Hit]:
        evidence: List[str] = []
        for rx in self._compiled:
            for m in rx.finditer(text):
                snippet = m.group(0).strip()
                if snippet and snippet not in evidence:
                    evidence.append(snippet)
        if self.custom is not None:
            for snippet in self.custom(text):
                if snippet not in evidence:
                    evidence.append(snippet)
        if not evidence:
            return None
        return Hit(self.id, self.category, self.weight, self.title, self.explanation,
                   evidence[:5])


# --- custom matchers -------------------------------------------------------
_URL_RE = re.compile(r"\b(?:https?://|www\.)[^\s<>\")]+", re.IGNORECASE)
_IP_URL_RE = re.compile(r"https?://\d{1,3}(?:\.\d{1,3}){3}", re.IGNORECASE)
_SHORTENERS = {"bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd",
               "buff.ly", "rebrand.ly", "cutt.ly", "shorturl.at"}
_RISKY_TLDS = {".xyz", ".top", ".click", ".live", ".online", ".info", ".link",
               ".zip", ".mov", ".cn", ".ru", ".tk", ".gq", ".work", ".loan"}
_BRANDS = ["paypal", "apple", "amazon", "netflix", "microsoft", "google",
           "facebook", "instagram", "whatsapp", "fedex", "ups", "usps", "dhl",
           "irs", "hmrc", "coinbase", "binance", "chase", "wellsfargo", "bankofamerica"]


def _hosts(text: str) -> List[str]:
    hosts = []
    for url in _URL_RE.findall(text):
        host = re.sub(r"^https?://", "", url, flags=re.IGNORECASE)
        host = host.split("/")[0].split("?")[0].lower()
        hosts.append(host)
    return hosts


def _match_shortener(text: str) -> List[str]:
    return [h for h in _hosts(text) if any(h == s or h.endswith("." + s) for s in _SHORTENERS)]


def _match_ip_url(text: str) -> List[str]:
    return [m.group(0) for m in _IP_URL_RE.finditer(text)]


def _match_risky_tld(text: str) -> List[str]:
    out = []
    for h in _hosts(text):
        bare = h.split(":")[0]
        for tld in _RISKY_TLDS:
            if bare.endswith(tld):
                out.append(h)
                break
    return out


def _match_lookalike_domain(text: str) -> List[str]:
    """A trusted brand name appearing as a sub-part of an untrusted domain.

    e.g. 'paypal.secure-login.com' or 'apple-id-verify.com' — the brand is in the
    host but the registered domain is not the brand's real one.
    """
    out = []
    for h in _hosts(text):
        bare = h.split(":")[0]
        labels = bare.split(".")
        if len(labels) < 2:
            continue
        registered = ".".join(labels[-2:])  # e.g. secure-login.com
        for brand in _BRANDS:
            in_host = brand in bare
            is_real = registered in (f"{brand}.com", f"{brand}.net", f"{brand}.org",
                                     f"{brand}.co.uk", f"{brand}.gov")
            if in_host and not is_real:
                out.append(h)
                break
    return out


def _match_punycode(text: str) -> List[str]:
    return [h for h in _hosts(text) if "xn--" in h]


# --- the rule set ----------------------------------------------------------
RULES: List[Rule] = [
    Rule("urgency", "Pressure", 18, "Creates false urgency",
         "Scammers rush you so you act before thinking. Legitimate organisations give you time.",
         patterns=[r"\b(act|respond|reply|confirm|verify|pay)\s+(now|immediately|within|in)\b",
                   r"\b(urgent|immediately|right away|asap|final notice|last warning)\b",
                   r"\bwithin \d+\s*(hours?|minutes?|days?)\b",
                   r"\byour (account|card|payment) will be (suspended|closed|blocked|deactivated)\b"]),
    Rule("threat", "Pressure", 16, "Threatens a consequence",
         "Threats of account closure, fines, arrest, or legal action are a classic pressure tactic.",
         patterns=[r"\b(suspend|deactivat|terminat|clos|lock|freez)\w* your (account|card)\b",
                   r"\byour (account|card)\b.{0,30}\b(suspend|deactivat|terminat|clos|lock|freez|block)\w*\b",
                   r"\b(arrest|lawsuit|legal action|court|fine|penalty|warrant)\b",
                   r"\byou (will|could) (lose|be charged|be fined|be arrested)\b"]),
    Rule("credentials", "Data theft", 22, "Asks for passwords / login / OTP",
         "No legitimate company asks you to send a password, PIN, or one-time code.",
         patterns=[r"\b(password|passcode|pin|otp|one[- ]?time (code|password)|2fa code|security code)\b",
                   r"\b(verify|confirm|update) your (identity|account|login|credentials)\b",
                   r"\b(login|log in|sign in) (here|now|to verify)\b"]),
    Rule("financial", "Money", 22, "Requests money or payment details",
         "Requests for gift cards, crypto, wire transfers, or card/bank details are top scam markers.",
         patterns=[r"\b(gift card|itunes card|google play card|steam card)\b",
                   r"\b(bitcoin|btc|ethereum|crypto(currency)?|usdt|wire transfer|western union|moneygram)\b",
                   r"\b(send|transfer|pay|deposit)\s+\$?\d[\d,]*\b",
                   r"\b(bank account|routing number|card number|cvv|sort code|iban)\b"]),
    Rule("prize", "Too good to be true", 16, "Promises a prize, refund, or windfall",
         "Unexpected winnings, refunds, inheritances, or 'you've been selected' offers are bait.",
         patterns=[r"\b(you('| ?ve| have)? (won|been selected|are a winner))\b",
                   r"\b(congratulations|claim your (prize|reward|refund|gift))\b",
                   r"\b(lottery|jackpot|inheritance|unclaimed (funds|money)|tax refund)\b",
                   r"\b(free|100% free) (iphone|gift|money|cash|vacation)\b"]),
    Rule("impersonation", "Impersonation", 14, "Impersonates a known organisation",
         "Messages claiming to be your bank, a delivery service, or a tax office are often spoofed.",
         patterns=[r"\b(your (bank|paypal|amazon|apple|netflix|microsoft) account)\b",
                   r"\b(irs|hmrc|social security|tax (office|department)|customs)\b",
                   r"\b(fedex|ups|usps|dhl|royal mail).{0,30}(package|parcel|delivery|shipment)\b",
                   r"\b(we ('| a)?re|this is) (your bank|the bank|customer (service|support))\b"]),
    Rule("delivery_fee", "Money", 14, "Asks a fee to release a package",
         "Couriers don't text you to pay a small 'redelivery' or 'customs' fee via a link.",
         patterns=[r"\b(pay|settle).{0,20}(customs|shipping|redelivery|handling) fee\b",
                   r"\bsmall fee\b.{0,30}(deliver|package|parcel)\b"]),
    Rule("shortener", "Suspicious link", 12, "Hides the destination with a link shortener",
         "Shortened links conceal where they really go. Hover or expand before clicking.",
         custom=_match_shortener),
    Rule("ip_url", "Suspicious link", 16, "Links to a raw IP address",
         "Real businesses use named domains, not bare IP addresses like http://192.168.x.x.",
         custom=_match_ip_url),
    Rule("lookalike", "Suspicious link", 20, "Look-alike / spoofed brand domain",
         "A trusted brand name appears in the address but the real domain isn't theirs.",
         custom=_match_lookalike_domain),
    Rule("risky_tld", "Suspicious link", 10, "Link uses a high-abuse domain ending",
         "Endings like .xyz, .top, .click, .zip are disproportionately used in scams.",
         custom=_match_risky_tld),
    Rule("punycode", "Suspicious link", 16, "Disguised (punycode) domain",
         "'xn--' domains can render look-alike characters to imitate a real site.",
         custom=_match_punycode),
    Rule("channel_switch", "Evasion", 12, "Pushes you to a private channel",
         "Moving you to WhatsApp/Telegram/personal email avoids monitoring and recovery.",
         patterns=[r"\b(contact|message|reach|text|add) (me|us) on (whatsapp|telegram|signal|wechat)\b",
                   r"\b(whatsapp|telegram).{0,15}(\+?\d[\d\s\-]{7,})\b"]),
    Rule("secrecy", "Evasion", 10, "Asks you to keep it secret",
         "'Don't tell anyone' / 'keep this confidential' isolates you from someone who'd spot the scam.",
         patterns=[r"\b(keep this|stay) (confidential|secret|between us)\b",
                   r"\bdon('| )?t (tell|inform|discuss with) (anyone|anybody|your)\b"]),
    Rule("generic_greeting", "Low-effort", 6, "Impersonal greeting",
         "Mass scams use 'Dear Customer/User' because they don't know your name.",
         patterns=[r"\bdear (customer|user|client|account holder|sir/madam|member)\b"]),
    Rule("attachment", "Suspicious link", 8, "Pushes an unexpected attachment",
         "Unexpected invoices, receipts, or 'documents' are a common malware delivery method.",
         patterns=[r"\b(open|see|review|download) (the )?(attached|attachment|invoice|receipt|document)\b",
                   r"\.(zip|rar|exe|scr|js|docm|xlsm)\b"]),
]
