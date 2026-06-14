import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr

from dota2tuned.ui.gradio_app import APP_CSS, build_app

demo = build_app()


if __name__ == "__main__":
    demo.launch(css=APP_CSS, theme=gr.themes.Soft())
