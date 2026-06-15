import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr

from dota2tuned.ui.gradio_app import APP_CSS, APP_HEAD, build_app, launch_app_kwargs

demo = build_app()


if __name__ == "__main__":
    # NOTE: the load JS is attached inside build_app via demo.load(); launch(js=)
    # is a no-op on page load in Gradio 6.18, so it is intentionally not passed.
    # app_kwargs wires the critical-head middleware that prevents the bare-element
    # flash (Gradio injects head= only after mount; see launch_app_kwargs).
    demo.launch(
        css=APP_CSS,
        head=APP_HEAD,
        theme=gr.themes.Soft(),
        app_kwargs=launch_app_kwargs(),
    )
