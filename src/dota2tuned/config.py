from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _secret_env(name: str) -> str | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    value = raw.strip()
    if not value or value.endswith("_xxx") or value in {"xxx", "changeme", "TODO"}:
        return None
    return value


@dataclass(frozen=True)
class Settings:
    hf_token: str | None = field(default_factory=lambda: _secret_env("HF_TOKEN"))
    hf_jobs_token: str | None = field(
        default_factory=lambda: _secret_env("HF_JOBS_TOKEN")
        or _secret_env("HF_UPLOAD_TOKEN")
        or _secret_env("HF_TOKEN")
    )
    hf_org: str = field(default_factory=lambda: os.getenv("HF_ORG", "build-small-hackathon"))
    hf_space_id: str = field(
        default_factory=lambda: os.getenv("HF_SPACE_ID", "build-small-hackathon/dota2tuned")
    )
    hf_model_repo_id: str = field(
        default_factory=lambda: os.getenv(
            "HF_MODEL_REPO_ID", "build-small-hackathon/dota2tuned-qwen3-4b-2507-lora"
        )
    )
    hf_dataset_repo_id: str = field(
        default_factory=lambda: os.getenv(
            "HF_DATASET_REPO_ID", "build-small-hackathon/dota2tuned-data"
        )
    )

    stratz_token: str | None = field(default_factory=lambda: _secret_env("STRATZ_TOKEN"))
    opendota_api_key: str | None = field(default_factory=lambda: _secret_env("OPENDOTA_API_KEY"))
    steam_api_key: str | None = field(default_factory=lambda: _secret_env("STEAM_API_KEY"))

    base_model_id: str = field(
        default_factory=lambda: os.getenv("BASE_MODEL_ID", "Qwen/Qwen3-4B-Instruct-2507")
    )
    fallback_base_model_id: str = field(
        default_factory=lambda: os.getenv("FALLBACK_BASE_MODEL_ID", "HuggingFaceTB/SmolLM3-3B")
    )
    training_flavor: str = field(default_factory=lambda: os.getenv("TRAINING_FLAVOR", "a100-large"))
    hf_job_timeout: str = field(default_factory=lambda: os.getenv("HF_JOB_TIMEOUT", "6h"))
    space_hardware: str = field(default_factory=lambda: os.getenv("SPACE_HARDWARE", "a10g-large"))
    sft_max_length: int = field(default_factory=lambda: int(os.getenv("SFT_MAX_LENGTH", "4096")))

    modal_enabled: bool = field(default_factory=lambda: _bool_env("MODAL_ENABLED", False))
    modal_app_name: str = field(default_factory=lambda: os.getenv("MODAL_APP_NAME", "dota2tuned"))
    modal_token_id: str | None = field(default_factory=lambda: _secret_env("MODAL_TOKEN_ID"))
    modal_token_secret: str | None = field(
        default_factory=lambda: _secret_env("MODAL_TOKEN_SECRET")
    )
    modal_train_gpu: str = field(
        default_factory=lambda: os.getenv("MODAL_TRAIN_GPU", "A100-80GB")
    )
    modal_train_timeout: int = field(
        default_factory=lambda: int(os.getenv("MODAL_TRAIN_TIMEOUT", str(6 * 60 * 60)))
    )
    modal_cache_volume: str = field(
        default_factory=lambda: os.getenv("MODAL_CACHE_VOLUME", "dota2tuned-hf-cache")
    )
    modal_output_volume: str = field(
        default_factory=lambda: os.getenv("MODAL_OUTPUT_VOLUME", "dota2tuned-outputs")
    )

    duckdb_path: Path = field(
        default_factory=lambda: Path(os.getenv("DUCKDB_PATH", "data/dota2tuned.duckdb"))
    )
    raw_data_dir: Path = field(default_factory=lambda: Path(os.getenv("RAW_DATA_DIR", "data/raw")))
    parquet_dir: Path = field(
        default_factory=lambda: Path(os.getenv("PARQUET_DIR", "data/parquet"))
    )
    rag_dir: Path = field(default_factory=lambda: Path(os.getenv("RAG_DIR", "data/rag")))
    model_dir: Path = field(default_factory=lambda: Path(os.getenv("MODEL_DIR", "data/models")))

    def ensure_dirs(self) -> None:
        for path in [
            self.duckdb_path.parent,
            self.raw_data_dir,
            self.parquet_dir,
            self.rag_dir,
            self.model_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
