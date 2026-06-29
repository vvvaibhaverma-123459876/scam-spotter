"""Gradio web app — the public, anyone-can-try interface.

Paste a suspicious message, get a calibrated risk score, the exact red flags that
fired (explainable), and concrete advice. Loads the DistilBERT classifier once at
startup and falls back to the torch-free heuristic if the model can't load.
"""
from __future__ import annotations

from functools import lru_cache

from .analyzer import Report, ScamAnalyzer

_EXAMPLES = [
    "URGENT: Your PayPal account has been limited. Verify within 24 hours at http://paypal.secure-verify.xyz or it will be permanently closed.",
    "Congratulations! You have WON a $1000 Amazon gift card. Claim your prize now: http://bit.ly/claim-reward",
    "Your FedEx package is held at customs. Pay a small redelivery fee of $2.99 here: http://fedex-redelivery.top/pay",
    "Hi, are we still on for lunch tomorrow at noon?",
    "Your Amazon order #112-9928 has shipped and will arrive Tuesday.",
]

_BADGE = {
    "Critical": ("#b71c1c", "🚨"),
    "High": ("#e53935", "⚠️"),
    "Medium": ("#fb8c00", "🟠"),
    "Low": ("#2e7d32", "✅"),
}


@lru_cache(maxsize=1)
def get_analyzer() -> ScamAnalyzer:
    return ScamAnalyzer()


def _verdict_md(report: Report) -> str:
    color, emoji = _BADGE.get(report.risk_level, ("#555", "•"))
    bar = "█" * round(report.risk_score / 5) + "░" * (20 - round(report.risk_score / 5))
    return (
        f"## {emoji} <span style='color:{color}'>{report.risk_level} risk — "
        f"{report.risk_score}/100</span>\n\n"
        f"`{bar}`\n\n"
        f"Model confidence this is a scam: **{report.ml_phishing_prob:.0%}** "
        f"_(via {report.classifier_source})_"
    )


def _flags_table(report: Report):
    if not report.hits:
        return [["—", "No scam signals detected", ""]]
    return [[h.category, h.title, ", ".join(h.evidence[:3])] for h in report.hits]


def analyze_message(text: str):
    report = get_analyzer().analyze(text or "")
    advice = "\n".join(f"- {a}" for a in report.advice)
    return _verdict_md(report), _flags_table(report), advice


def build_demo():
    import gradio as gr

    # `theme` is accepted by Blocks on Gradio 4/5 and by launch() on 6+, so we
    # set it where the installed version expects it to avoid a deprecation warning.
    blocks_kwargs = {"title": "Scam Spotter"}
    if int(gr.__version__.split(".")[0]) < 6:
        blocks_kwargs["theme"] = gr.themes.Soft()

    with gr.Blocks(**blocks_kwargs) as demo:
        gr.Markdown(
            "# 🛡️ Scam Spotter\n"
            "Paste a suspicious **text, email, or DM** and find out if it's likely a scam — "
            "and *why*. Powered by a DistilBERT phishing classifier plus an explainable "
            "red-flag engine. Nothing is stored.\n"
        )
        with gr.Row():
            with gr.Column(scale=3):
                inp = gr.Textbox(label="Message to check", lines=7,
                                 placeholder="Paste the message here…")
                btn = gr.Button("Check this message", variant="primary")
                gr.Examples(_EXAMPLES, inputs=inp, label="Try an example")
            with gr.Column(scale=2):
                verdict = gr.Markdown()
                advice = gr.Markdown(label="What to do")
        flags = gr.Dataframe(headers=["Category", "Red flag", "What triggered it"],
                             label="Why — explainable red flags", wrap=True,
                             interactive=False)
        gr.Markdown(
            "<sub>Educational tool. Not legal or security advice — when in doubt, "
            "contact the organisation through its official website or app.</sub>"
        )
        btn.click(analyze_message, inputs=inp, outputs=[verdict, flags, advice])
        inp.submit(analyze_message, inputs=inp, outputs=[verdict, flags, advice])
    return demo


demo = None


def main() -> None:
    build_demo().launch()


if __name__ == "__main__":
    main()
