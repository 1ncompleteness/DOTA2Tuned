from __future__ import annotations

from pathlib import Path

from dota2tuned.config import Settings

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
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

model_id = "{model_id}"
dataset_source = "{dataset_source}"
output_repo = "{output_repo}"

if dataset_source.endswith(".jsonl"):
    dataset = load_dataset("json", data_files=dataset_source, split="train")
else:
    dataset = load_dataset(dataset_source, split="train")
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
    packing=True,
    push_to_hub=True,
    hub_model_id=output_repo,
)
trainer = SFTTrainer(
    model=model_id,
    args=args,
    train_dataset=dataset,
    peft_config=peft_config,
)
trainer.train()
trainer.push_to_hub(output_repo)
"""


def write_train_script(settings: Settings, dataset_source: str | Path) -> Path:
    script_path = settings.model_dir / "train_sft_job.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(
        TRAIN_SCRIPT.format(
            model_id=settings.base_model_id,
            dataset_source=str(dataset_source),
            output_repo=settings.hf_model_repo_id,
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
        from huggingface_hub import run_uv_job
    except ImportError as exc:
        raise RuntimeError("huggingface_hub with Jobs support is required for run_uv_job.") from exc
    job = run_uv_job(
        str(script_path),
        dependencies=["datasets", "transformers", "trl", "peft", "bitsandbytes", "accelerate"],
        secrets={"HF_TOKEN": settings.hf_token},
        flavor=settings.training_flavor,
    )
    return str(job)
