from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from dota2tuned.config import get_settings

try:
    import modal
except ImportError:  # pragma: no cover - optional local dependency
    modal = None


settings = get_settings()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REMOTE_ROOT = Path("/app")
REMOTE_DATA = REMOTE_ROOT / "data"
REMOTE_CACHE = Path("/cache")
REMOTE_OUTPUTS = Path("/outputs")

CORE_DEPS = [
    "duckdb>=1.1.0",
    "fastapi[standard]>=0.115.0",
    "gradio>=5.0.0",
    "httpx>=0.27.0",
    "huggingface-hub>=1.0.0",
    "joblib>=1.4.0",
    "numpy>=1.26.0",
    "polars>=1.0.0",
    "pyarrow>=16.0.0",
    "pydantic>=2.8.0",
    "python-dotenv>=1.0.0",
    "scikit-learn>=1.5.0",
    "tenacity>=8.3.0",
]

TRAIN_DEPS = [
    *CORE_DEPS,
    "accelerate>=1.0.0",
    "bitsandbytes>=0.44.0",
    "datasets>=3.0.0",
    "hf-transfer>=0.1.9",
    "peft>=0.13.0",
    "torch>=2.4.0",
    "transformers>=4.51.0",
    "trl>=0.25.0",
]

REMOTE_ENV = {
    "PYTHONPATH": str(REMOTE_ROOT),
    "MODAL_ENABLED": "1",
    "MODAL_APP_NAME": settings.modal_app_name,
    "PARQUET_DIR": str(REMOTE_DATA / "parquet"),
    "RAG_DIR": str(REMOTE_DATA / "rag"),
    "MODEL_DIR": str(REMOTE_DATA / "models"),
    "RAW_DATA_DIR": str(REMOTE_DATA / "raw"),
    "DUCKDB_PATH": str(REMOTE_DATA / "dota2tuned.duckdb"),
    "BASE_MODEL_ID": settings.base_model_id,
    "HF_MODEL_REPO_ID": settings.hf_model_repo_id,
    "HF_DATASET_REPO_ID": settings.hf_dataset_repo_id,
    "SFT_MAX_LENGTH": str(settings.sft_max_length),
    "HF_HOME": str(REMOTE_CACHE / "huggingface"),
    "HF_HUB_CACHE": str(REMOTE_CACHE / "huggingface" / "hub"),
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "TQDM_DISABLE": "1",
    "TRANSFORMERS_VERBOSITY": "warning",
}


def _local_dir(name: str) -> Path:
    return PROJECT_ROOT / "data" / name


def _runtime_secret_dict() -> dict[str, str]:
    values = {
        "HF_TOKEN": settings.hf_token,
        "HUGGINGFACE_HUB_TOKEN": settings.hf_token,
        "STRATZ_TOKEN": settings.stratz_token,
        "OPENDOTA_API_KEY": settings.opendota_api_key,
        "STEAM_API_KEY": settings.steam_api_key,
    }
    return {key: value for key, value in values.items() if value}


def _add_project_files(image: modal.Image) -> modal.Image:
    image = image.add_local_dir(
        PROJECT_ROOT / "src" / "dota2tuned",
        str(REMOTE_ROOT / "dota2tuned"),
        copy=True,
    )
    for data_dir in ["parquet", "rag", "models"]:
        local_path = _local_dir(data_dir)
        if local_path.exists():
            image = image.add_local_dir(local_path, str(REMOTE_DATA / data_dir), copy=True)
    return image


if modal is not None and settings.modal_enabled:
    app = modal.App(settings.modal_app_name)
    runtime_secret = modal.Secret.from_dict(_runtime_secret_dict())
    cache_volume = modal.Volume.from_name(settings.modal_cache_volume, create_if_missing=True)
    output_volume = modal.Volume.from_name(settings.modal_output_volume, create_if_missing=True)

    web_image = _add_project_files(
        modal.Image.debian_slim(python_version="3.12").uv_pip_install(*CORE_DEPS).env(REMOTE_ENV)
    )
    train_image = _add_project_files(
        modal.Image.debian_slim(python_version="3.12").uv_pip_install(*TRAIN_DEPS).env(REMOTE_ENV)
    )

    @app.function(image=web_image, secrets=[runtime_secret], timeout=900)
    def remote_smoke() -> dict[str, object]:
        from dota2tuned.ui.gradio_app import build_app

        demo = build_app()
        return {
            "app": type(demo).__name__,
            "blocks": len(demo.blocks),
            "callbacks": len(demo.fns),
            "parquet_files": len(list((REMOTE_DATA / "parquet").glob("*.parquet"))),
            "rag_index": (REMOTE_DATA / "rag" / "tfidf.joblib").exists(),
            "sft_examples": (REMOTE_DATA / "models" / "sft_examples.jsonl").exists(),
        }

    @app.function(image=web_image, secrets=[runtime_secret], timeout=900, max_containers=1)
    @modal.concurrent(max_inputs=100)
    @modal.asgi_app()
    def ui():
        from fastapi import FastAPI
        from gradio.routes import mount_gradio_app

        from dota2tuned.ui.gradio_app import build_app

        return mount_gradio_app(app=FastAPI(), blocks=build_app(), path="/")

    @app.function(
        image=train_image,
        gpu=settings.modal_train_gpu,
        secrets=[runtime_secret],
        volumes={REMOTE_CACHE: cache_volume, REMOTE_OUTPUTS: output_volume},
        timeout=settings.modal_train_timeout,
    )
    def train_sft(
        dataset_source: str = str(REMOTE_DATA / "models" / "sft_examples.jsonl"),
    ) -> dict[str, object]:
        from dota2tuned.config import Settings
        from dota2tuned.finetune import write_train_script

        if not os.environ.get("HF_TOKEN"):
            raise RuntimeError("HF_TOKEN is required in the Modal runtime secret.")
        if not Path(dataset_source).exists() and not dataset_source.startswith(
            settings.hf_dataset_repo_id
        ):
            raise RuntimeError(f"SFT dataset not found: {dataset_source}")

        REMOTE_OUTPUTS.mkdir(parents=True, exist_ok=True)
        train_settings = Settings(
            hf_token=os.environ.get("HF_TOKEN"),
            hf_model_repo_id=os.environ["HF_MODEL_REPO_ID"],
            hf_dataset_repo_id=os.environ["HF_DATASET_REPO_ID"],
            base_model_id=os.environ["BASE_MODEL_ID"],
            sft_max_length=int(os.environ["SFT_MAX_LENGTH"]),
            raw_data_dir=REMOTE_DATA / "raw",
            parquet_dir=REMOTE_DATA / "parquet",
            rag_dir=REMOTE_DATA / "rag",
            model_dir=REMOTE_OUTPUTS,
            duckdb_path=REMOTE_DATA / "dota2tuned.duckdb",
        )
        script_path = write_train_script(train_settings, dataset_source)
        subprocess.run([sys.executable, str(script_path)], check=True)
        return {
            "status": "ok",
            "base_model": train_settings.base_model_id,
            "output_repo": train_settings.hf_model_repo_id,
            "dataset_source": dataset_source,
            "script": str(script_path),
        }
else:
    app = None


def modal_available() -> bool:
    return app is not None
