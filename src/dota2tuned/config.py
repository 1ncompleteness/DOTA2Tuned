from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    hf_token: str | None = os.getenv("HF_TOKEN") or None
    hf_org: str = os.getenv("HF_ORG", "build-small-hackathon")
    hf_space_id: str = os.getenv("HF_SPACE_ID", "build-small-hackathon/dota2tuned")
    hf_model_repo_id: str = os.getenv(
        "HF_MODEL_REPO_ID", "build-small-hackathon/dota2tuned-qwen3-4b-2507-lora"
    )
    hf_dataset_repo_id: str = os.getenv(
        "HF_DATASET_REPO_ID", "build-small-hackathon/dota2tuned-data"
    )

    stratz_token: str | None = os.getenv("STRATZ_TOKEN") or None
    opendota_api_key: str | None = os.getenv("OPENDOTA_API_KEY") or None
    steam_api_key: str | None = os.getenv("STEAM_API_KEY") or None

    base_model_id: str = os.getenv("BASE_MODEL_ID", "Qwen/Qwen3-4B-Instruct-2507")
    fallback_base_model_id: str = os.getenv("FALLBACK_BASE_MODEL_ID", "HuggingFaceTB/SmolLM3-3B")
    training_flavor: str = os.getenv("TRAINING_FLAVOR", "a10g-large")
    hf_job_timeout: str = os.getenv("HF_JOB_TIMEOUT", "6h")
    space_hardware: str = os.getenv("SPACE_HARDWARE", "a10g-large")
    sft_max_length: int = int(os.getenv("SFT_MAX_LENGTH", "4096"))

    modal_enabled: bool = _bool_env("MODAL_ENABLED", False)
    modal_app_name: str = os.getenv("MODAL_APP_NAME", "dota2tuned")
    modal_token_id: str | None = os.getenv("MODAL_TOKEN_ID") or None
    modal_token_secret: str | None = os.getenv("MODAL_TOKEN_SECRET") or None

    duckdb_path: Path = Path(os.getenv("DUCKDB_PATH", "data/dota2tuned.duckdb"))
    raw_data_dir: Path = Path(os.getenv("RAW_DATA_DIR", "data/raw"))
    parquet_dir: Path = Path(os.getenv("PARQUET_DIR", "data/parquet"))
    rag_dir: Path = Path(os.getenv("RAG_DIR", "data/rag"))
    model_dir: Path = Path(os.getenv("MODEL_DIR", "data/models"))

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
