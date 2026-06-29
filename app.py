"""Hugging Face Space entry point.

Builds the Gradio demo and launches it. On Spaces, Gradio binds 0.0.0.0:7860
automatically; locally this serves on http://127.0.0.1:7860.
"""
from scam_spotter.app import build_demo

demo = build_demo()

if __name__ == "__main__":
    demo.launch()
