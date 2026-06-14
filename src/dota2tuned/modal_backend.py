from __future__ import annotations

from dota2tuned.config import get_settings

settings = get_settings()


if settings.modal_enabled:
    import modal

    image = modal.Image.debian_slim(python_version="3.11").uv_pip_install(
        "transformers", "torch", "accelerate", "huggingface_hub"
    )
    app = modal.App(settings.modal_app_name)

    @app.function(image=image, gpu="A10G", timeout=900)
    def gpu_echo(prompt: str) -> str:
        return prompt
else:
    app = None


def modal_available() -> bool:
    return app is not None
