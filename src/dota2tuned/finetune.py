from __future__ import annotations

from pathlib import Path
from typing import Any

from dota2tuned.config import Settings

JOB_DEPENDENCIES = [
    "datasets",
    "transformers",
    "trl",
    "peft",
    "bitsandbytes",
    "accelerate",
    "huggingface_hub",
    "hf-transfer",
]

JOB_ENV = {
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "TQDM_DISABLE": "1",
    "TRANSFORMERS_VERBOSITY": "warning",
    "UV_NO_PROGRESS": "1",
}

TRAIN_SCRIPT = """# /// script
# dependencies = [
#   "datasets",
#   "transformers",
#   "trl",
#   "peft",
#   "bitsandbytes",
#   "accelerate",
#   "huggingface_hub",
# ]
# ///
import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

model_id = "{model_id}"
dataset_source = "{dataset_source}"
output_repo = "{output_repo}"
max_length = {max_length}

if dataset_source.endswith(".jsonl"):
    dataset = load_dataset("json", data_files=dataset_source, split="train")
else:
    dataset = load_dataset(dataset_source, split="train")

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=32,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)
args = SFTConfig(
    output_dir="dota2tuned-sft",
    num_train_epochs=1,
    learning_rate=2e-4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    max_length=max_length,
    packing=True,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    bf16=True,
    logging_steps=5,
    save_strategy="epoch",
    push_to_hub=True,
    hub_model_id=output_repo,
)
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
)
trainer.train()
trainer.push_to_hub(output_repo)
"""


def _token_permissions(whoami: dict[str, Any], namespace: str) -> set[str] | None:
    token = ((whoami.get("auth") or {}).get("accessToken") or {})
    role = token.get("role")
    if role and role != "fineGrained":
        return None

    fine_grained = token.get("fineGrained")
    if not isinstance(fine_grained, dict):
        return None

    permissions = set(fine_grained.get("global") or [])
    for scope in fine_grained.get("scoped") or []:
        entity = scope.get("entity") or {}
        if entity.get("name") == namespace:
            permissions.update(scope.get("permissions") or [])
    return permissions


def validate_hf_jobs_access(settings: Settings) -> dict[str, str]:
    if not settings.hf_token:
        raise RuntimeError("HF_TOKEN is required to launch a Hugging Face Job.")
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub with Jobs support is required.") from exc

    api = HfApi(token=settings.hf_token)
    whoami = api.whoami()
    permissions = _token_permissions(whoami, settings.hf_org)
    if permissions is not None and "job.write" not in permissions:
        raise RuntimeError(
            "HF_TOKEN is authenticated but cannot launch Hugging Face Jobs. "
            f"Add `job.write` to the token scope for namespace `{settings.hf_org}` "
            "or use a token that can start/manage Jobs for that organization. "
            f"Current parsed permissions: {sorted(permissions)}"
        )

    hardware_names = {hardware.name for hardware in api.list_jobs_hardware()}
    if settings.training_flavor not in hardware_names:
        raise RuntimeError(
            f"TRAINING_FLAVOR={settings.training_flavor!r} is not available for Jobs. "
            f"Available flavors include: {', '.join(sorted(hardware_names))}"
        )

    return {
        "namespace": settings.hf_org,
        "flavor": settings.training_flavor,
        "user": str(whoami.get("name") or "unknown"),
    }


def write_train_script(settings: Settings, dataset_source: str | Path) -> Path:
    script_path = settings.model_dir / "train_sft_job.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(
        TRAIN_SCRIPT.format(
            model_id=settings.base_model_id,
            dataset_source=str(dataset_source),
            output_repo=settings.hf_model_repo_id,
            max_length=settings.sft_max_length,
        )
    )
    return script_path


def upload_sft_dataset(settings: Settings, dataset_path: Path) -> str:
    if not settings.hf_token:
        raise RuntimeError("HF_TOKEN is required to upload the SFT dataset.")
    if not dataset_path.exists():
        raise RuntimeError(f"SFT dataset does not exist: {dataset_path}")
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to upload datasets.") from exc

    api = HfApi(token=settings.hf_token)
    api.create_repo(
        repo_id=settings.hf_dataset_repo_id,
        repo_type="dataset",
        exist_ok=True,
    )
    api.upload_file(
        path_or_fileobj=str(dataset_path),
        path_in_repo="sft_examples.jsonl",
        repo_id=settings.hf_dataset_repo_id,
        repo_type="dataset",
    )
    return settings.hf_dataset_repo_id


def launch_hf_job(settings: Settings, script_path: Path) -> str:
    if not settings.hf_token:
        raise RuntimeError("HF_TOKEN is required to launch a Hugging Face Job.")
    try:
        from huggingface_hub import HfApi, run_uv_job
    except ImportError as exc:
        raise RuntimeError("huggingface_hub with Jobs support is required for run_uv_job.") from exc
    HfApi(token=settings.hf_token).create_repo(
        repo_id=settings.hf_model_repo_id,
        repo_type="model",
        exist_ok=True,
    )
    job = run_uv_job(
        str(script_path),
        dependencies=JOB_DEPENDENCIES,
        env=JOB_ENV,
        secrets={
            "HF_TOKEN": settings.hf_token,
            "HUGGINGFACE_HUB_TOKEN": settings.hf_token,
        },
        flavor=settings.training_flavor,
        timeout=settings.hf_job_timeout,
        namespace=settings.hf_org,
        token=settings.hf_token,
    )
    return str(job)
