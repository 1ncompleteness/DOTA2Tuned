import os
import sys
from pathlib import Path

# Force client-side rendering before Gradio is imported/launched. HF Spaces sets
# GRADIO_SSR_MODE=True, which runs a Node SSR proxy that server-renders the bare
# app and bypasses our critical-head loader/middleware (flash of bare elements).
# Overriding the env here keeps rendering client-side so the loader covers startup.
os.environ["GRADIO_SSR_MODE"] = "False"

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dota2tuned.ui.runtime_hooks import (  # noqa: E402
    close_idle_main_event_loop,
    gradio_launch_runtime_kwargs,
    install_quiet_unraisablehook,
)

# Install before importing Gradio. Gradio 6 on HF Spaces can create orphaned
# asyncio loops during import/startup; Python 3.12 reports those as noisy
# unraisable "Invalid file descriptor: -1" tracebacks.
install_quiet_unraisablehook()

import gradio as gr  # noqa: E402

from dota2tuned.ui.gradio_app import (  # noqa: E402
    APP_CSS,
    APP_HEAD,
    build_app,
    launch_app_kwargs,
)

demo = build_app()
close_idle_main_event_loop()


if __name__ == "__main__":
    # NOTE: the load JS is attached inside build_app via demo.load(); launch(js=)
    # is a no-op on page load in Gradio 6.18, so it is intentionally not passed.
    # app_kwargs wires the critical-head middleware that prevents the bare-element
    # flash (Gradio injects head= only after mount; see launch_app_kwargs).
    # ssr_mode=False: HF Spaces auto-enables SSR (Node proxy), which server-renders
    # the bare app and bypasses both our head= loader and the Python middleware.
    # Client-side rendering lets the critical-head middleware hide the app until ready.
    demo.launch(
        css=APP_CSS,
        head=APP_HEAD,
        theme=gr.themes.Soft(),
        ssr_mode=False,
        **gradio_launch_runtime_kwargs(),
        app_kwargs=launch_app_kwargs(),
    )
