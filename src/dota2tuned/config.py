from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from dota2tuned.model_profiles import apply_profile_env_overrides, resolve_model_profile

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


def _active_profile():
    return apply_profile_env_overrides(resolve_model_profile(os.getenv("MODEL_PROFILE")))


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
    model_profile: str = field(default_factory=lambda: os.getenv("MODEL_PROFILE", "qwen3_4b_2507"))
    model_profiles_path: Path | None = field(
        default_factory=lambda: Path(os.environ["MODEL_PROFILES_PATH"])
        if os.getenv("MODEL_PROFILES_PATH")
        else None
    )
    hf_model_repo_id: str = field(
        default_factory=lambda: _active_profile().hf_model_repo_id
    )
    hf_dataset_repo_id: str = field(
        default_factory=lambda: os.getenv(
            "HF_DATASET_REPO_ID", "build-small-hackathon/dota2tuned-data"
        )
    )

    stratz_token: str | None = field(default_factory=lambda: _secret_env("STRATZ_TOKEN"))
    opendota_api_key: str | None = field(default_factory=lambda: _secret_env("OPENDOTA_API_KEY"))
    steam_api_key: str | None = field(default_factory=lambda: _secret_env("STEAM_API_KEY"))

    base_model_id: str = field(default_factory=lambda: _active_profile().base_model_id)
    fallback_base_model_id: str = field(
        default_factory=lambda: os.getenv("FALLBACK_BASE_MODEL_ID", "HuggingFaceTB/SmolLM3-3B")
    )
    training_flavor: str = field(default_factory=lambda: os.getenv("TRAINING_FLAVOR", "a100-large"))
    hf_job_timeout: str = field(default_factory=lambda: os.getenv("HF_JOB_TIMEOUT", "6h"))
    space_hardware: str = field(default_factory=lambda: os.getenv("SPACE_HARDWARE", "a10g-large"))
    sft_max_length: int = field(default_factory=lambda: _active_profile().sft_max_length)
    lora_r: int = field(default_factory=lambda: _active_profile().lora_r)
    lora_alpha: int = field(default_factory=lambda: _active_profile().lora_alpha)
    lora_dropout: float = field(default_factory=lambda: _active_profile().lora_dropout)
    lora_target_modules: str = field(
        default_factory=lambda: _active_profile().lora_target_modules
    )
    sft_learning_rate: float = field(default_factory=lambda: _active_profile().sft_learning_rate)
    sft_epochs: float = field(default_factory=lambda: _active_profile().sft_epochs)
    sft_batch_size: int = field(default_factory=lambda: _active_profile().sft_batch_size)
    sft_grad_accum: int = field(default_factory=lambda: _active_profile().sft_grad_accum)
    model_load_in_4bit: bool = field(default_factory=lambda: _active_profile().model_load_in_4bit)
    model_torch_dtype: str = field(default_factory=lambda: _active_profile().model_torch_dtype)

    modal_enabled: bool = field(default_factory=lambda: _bool_env("MODAL_ENABLED", False))
    modal_app_name: str = field(default_factory=lambda: os.getenv("MODAL_APP_NAME", "dota2tuned"))
    modal_token_id: str | None = field(default_factory=lambda: _secret_env("MODAL_TOKEN_ID"))
    modal_token_secret: str | None = field(
        default_factory=lambda: _secret_env("MODAL_TOKEN_SECRET")
    )
    modal_train_gpu: str = field(default_factory=lambda: _active_profile().modal_train_gpu)
    modal_train_timeout: int = field(
        default_factory=lambda: _active_profile().modal_train_timeout
    )
    modal_infer_gpu: str = field(default_factory=lambda: _active_profile().modal_infer_gpu)
    modal_infer_timeout: int = field(
        default_factory=lambda: _active_profile().modal_infer_timeout
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
