from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelProfile:
    key: str
    label: str
    base_model_id: str
    hf_model_repo_id: str
    description: str
    sft_max_length: int = 4096
    modal_train_gpu: str = "A100-80GB"
    modal_infer_gpu: str = "A100-80GB"
    modal_train_timeout: int = 6 * 60 * 60
    modal_infer_timeout: int = 900
    lora_r: int = 32
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: str = "all-linear"
    sft_learning_rate: float = 2e-4
    sft_epochs: float = 1.0
    sft_batch_size: int = 1
    sft_grad_accum: int = 8
    model_load_in_4bit: bool = True
    model_torch_dtype: str = "bfloat16"


DEFAULT_MODEL_PROFILE = "qwen3_4b_2507"

DEFAULT_MODEL_PROFILES: dict[str, ModelProfile] = {
    "qwen3_4b_2507": ModelProfile(
        key="qwen3_4b_2507",
        label="Qwen3 4B Tiny",
        base_model_id="Qwen/Qwen3-4B-Instruct-2507",
        hf_model_repo_id="build-small-hackathon/dota2tuned-qwen3-4b-2507-lora",
        description="Fast baseline adapter for demos and fallback inference.",
        modal_infer_gpu="A10G",
    ),
    "qwen3_30b_a3b_2507": ModelProfile(
        key="qwen3_30b_a3b_2507",
        label="Qwen3 30B-A3B Quality",
        base_model_id="Qwen/Qwen3-30B-A3B-Instruct-2507",
        hf_model_repo_id="build-small-hackathon/dota2tuned-qwen3-30b-a3b-2507-lora",
        description="Primary quality model under the 32B parameter cap.",
        sft_max_length=4096,
        modal_train_gpu="H200",
        modal_infer_gpu="H200",
        lora_r=32,
        lora_alpha=16,
        lora_dropout=0.0,
        lora_target_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        sft_learning_rate=1e-4,
        sft_grad_accum=16,
    ),
    "minicpm4_1_8b": ModelProfile(
        key="minicpm4_1_8b",
        label="MiniCPM4.1 8B Balanced",
        base_model_id="openbmb/MiniCPM4.1-8B",
        hf_model_repo_id="build-small-hackathon/dota2tuned-minicpm4-1-8b-lora",
        description="OpenBMB-path model for a distinct hackathon submission angle.",
        sft_max_length=8192,
        lora_r=32,
        lora_alpha=16,
        sft_learning_rate=1.5e-4,
        sft_grad_accum=8,
    ),
}


def _bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _profile_path() -> Path | None:
    raw = os.getenv("MODEL_PROFILES_PATH")
    if not raw:
        return None
    return Path(raw)


def load_model_profiles(path: Path | None = None) -> dict[str, ModelProfile]:
    profiles = dict(DEFAULT_MODEL_PROFILES)
    path = path or _profile_path()
    if not path or not path.exists():
        return profiles
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        return profiles
    for key, raw in payload.items():
        if not isinstance(raw, dict):
            continue
        base = profiles.get(key) or ModelProfile(
            key=key,
            label=str(raw.get("label") or key),
            base_model_id=str(raw.get("base_model_id") or ""),
            hf_model_repo_id=str(raw.get("hf_model_repo_id") or ""),
            description=str(raw.get("description") or ""),
        )
        profiles[key] = replace(
            base,
            **{field: value for field, value in raw.items() if hasattr(base, field)},
        )
    return profiles


def resolve_model_profile(key: str | None = None) -> ModelProfile:
    profiles = load_model_profiles()
    selected = key or os.getenv("MODEL_PROFILE") or DEFAULT_MODEL_PROFILE
    return profiles.get(selected) or profiles[DEFAULT_MODEL_PROFILE]


def profile_choices() -> list[tuple[str, str]]:
    return [(profile.label, key) for key, profile in load_model_profiles().items()]


def apply_profile_env_overrides(profile: ModelProfile) -> ModelProfile:
    return replace(
        profile,
        base_model_id=os.getenv("BASE_MODEL_ID", profile.base_model_id),
        hf_model_repo_id=os.getenv("HF_MODEL_REPO_ID", profile.hf_model_repo_id),
        sft_max_length=int(os.getenv("SFT_MAX_LENGTH", str(profile.sft_max_length))),
        modal_train_gpu=os.getenv("MODAL_TRAIN_GPU", profile.modal_train_gpu),
        modal_infer_gpu=os.getenv("MODAL_INFER_GPU", profile.modal_infer_gpu),
        modal_train_timeout=int(
            os.getenv("MODAL_TRAIN_TIMEOUT", str(profile.modal_train_timeout))
        ),
        modal_infer_timeout=int(
            os.getenv("MODAL_INFER_TIMEOUT", str(profile.modal_infer_timeout))
        ),
        lora_r=int(os.getenv("LORA_R", str(profile.lora_r))),
        lora_alpha=int(os.getenv("LORA_ALPHA", str(profile.lora_alpha))),
        lora_dropout=float(os.getenv("LORA_DROPOUT", str(profile.lora_dropout))),
        lora_target_modules=os.getenv("LORA_TARGET_MODULES", profile.lora_target_modules),
        sft_learning_rate=float(
            os.getenv("SFT_LEARNING_RATE", str(profile.sft_learning_rate))
        ),
        sft_epochs=float(os.getenv("SFT_EPOCHS", str(profile.sft_epochs))),
        sft_batch_size=int(os.getenv("SFT_BATCH_SIZE", str(profile.sft_batch_size))),
        sft_grad_accum=int(os.getenv("SFT_GRAD_ACCUM", str(profile.sft_grad_accum))),
        model_load_in_4bit=_bool(
            os.getenv("MODEL_LOAD_IN_4BIT"), profile.model_load_in_4bit
        ),
        model_torch_dtype=os.getenv("MODEL_TORCH_DTYPE", profile.model_torch_dtype),
    )
