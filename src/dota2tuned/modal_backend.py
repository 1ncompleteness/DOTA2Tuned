from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from dota2tuned.config import get_settings
from dota2tuned.model_profiles import resolve_model_profile

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
BALANCED_PROFILE_KEY = "minicpm4_1_8b"
QUALITY_PROFILE_KEY = "qwen3_30b_a3b_2507"
QUALITY_INFER_FUNCTION = "generate_answer_quality"
DEFAULT_INFER_FUNCTION = "generate_answer"
THINKING_TOKEN_LIMIT = 1536
NON_THINKING_TOKEN_LIMIT = 768
THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)

CORE_DEPS = [
    "duckdb>=1.5.3",
    "fastapi[standard]>=0.137.0",
    "gradio>=6.18.0",
    "httpx>=0.28.1",
    "huggingface-hub>=1.19.0",
    "joblib>=1.5.3",
    "numpy>=2.4.6",
    "polars>=1.41.2",
    "pyarrow>=24.0.0",
    "pydantic>=2.12.5,<2.13.0",
    "python-dotenv>=1.2.2",
    "scikit-learn>=1.9.0",
    "tenacity>=9.1.4",
]

TRAIN_DEPS = [
    *CORE_DEPS,
    "accelerate>=1.14.0",
    "bitsandbytes>=0.49.2",
    "datasets>=5.0.0",
    "hf-transfer>=0.1.9",
    "peft>=0.19.1",
    "torch>=2.12.0",
    "transformers>=5.12.0",
    "trl>=1.6.0",
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
    "MODEL_PROFILE": settings.model_profile,
    "BASE_MODEL_ID": settings.base_model_id,
    "HF_MODEL_REPO_ID": settings.hf_model_repo_id,
    "HF_DATASET_REPO_ID": settings.hf_dataset_repo_id,
    "SFT_MAX_LENGTH": str(settings.sft_max_length),
    "HF_HOME": str(REMOTE_CACHE / "huggingface"),
    "HF_HUB_CACHE": str(REMOTE_CACHE / "huggingface" / "hub"),
    "HF_XET_HIGH_PERFORMANCE": "1",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "TQDM_DISABLE": "1",
    "TRANSFORMERS_VERBOSITY": "warning",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
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


def modal_infer_function_name(profile: str | None = None) -> str:
    selected = resolve_model_profile(profile)
    if selected.key == QUALITY_PROFILE_KEY:
        return QUALITY_INFER_FUNCTION
    return DEFAULT_INFER_FUNCTION


def _looks_malformed_answer(answer: str) -> bool:
    text = " ".join(str(answer or "").split())
    if not text:
        return True
    if len(text) < 12:
        return True
    if text.count("\\") >= 6 or text.count("�") >= 2:
        return True
    nonspace = [char for char in text if not char.isspace()]
    if not nonspace:
        return True
    alpha_numeric = sum(char.isalnum() for char in nonspace)
    if alpha_numeric / len(nonspace) < 0.25:
        return True
    stripped_words = [
        word.strip(".,;:!?*#`\"'()[]{}")
        for word in text.split()
    ]
    word_like = [
        word
        for word in stripped_words
        if len(word) >= 3 and any(char.isalpha() for char in word)
    ]
    normalized_words = [word.lower() for word in word_like]
    if len(normalized_words) >= 8:
        counts = {
            word: normalized_words.count(word)
            for word in set(normalized_words)
        }
        stems = [word[:5] for word in normalized_words]
        stem_counts = {stem: stems.count(stem) for stem in set(stems)}
        if max(counts.values()) / len(normalized_words) > 0.28:
            return True
        if max(stem_counts.values()) / len(stems) > 0.45:
            return True
    if len(text) < 120 and len(word_like) < 5:
        return True
    marker_count = text.count("##") + text.count("**") + text.count("\\")
    if marker_count >= 3 and len(word_like) < 8:
        return True
    return len(set(nonspace)) <= 4 and len(nonspace) >= 24


def _evidence_fallback_answer(question: str, context: str) -> str:
    evidence_lines = [
        " ".join(line.split())
        for line in str(context or "").splitlines()
        if line.strip()
    ]
    evidence = " ".join(evidence_lines)[:900].strip()
    if not evidence:
        evidence = "No retrieved evidence was supplied for this question."
    return (
        "Based on the retrieved evidence, use the most directly supported draft or "
        "build choice rather than inventing outside context.\n\n"
        f"Question: {question.strip()}\n\n"
        f"Evidence summary: {evidence}\n\n"
        "Recommendation: prefer the hero, item, or draft choice with the strongest "
        "role fit and current-patch evidence in the retrieved context. Caveat: treat "
        "this as evidence-grounded guidance, not a guaranteed match outcome."
    )


def _strip_reasoning_blocks(answer: str) -> str:
    text = str(answer or "")
    text = THINK_RE.sub("", text)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    return text.strip()


def _install_peft_weight_converter_compat() -> bool:
    try:
        import inspect

        from transformers.core_model_loading import WeightConverter
    except Exception:
        return False

    if "distributed_operation" in inspect.signature(WeightConverter.__init__).parameters:
        return False
    if getattr(WeightConverter.__init__, "_dota2tuned_compat", False):
        return True

    original_init = WeightConverter.__init__

    def _compat_init(
        self,
        source_patterns,
        target_patterns,
        operations,
        distributed_operation=None,
        quantization_operation=None,
        **_kwargs,
    ):
        original_init(self, source_patterns, target_patterns, operations)
        self.distributed_operation = distributed_operation
        self.quantization_operation = quantization_operation

    _compat_init._dota2tuned_compat = True
    WeightConverter.__init__ = _compat_init
    return True


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
    _MODEL_CACHE: dict[str, object] = {}
    quality_train_profile = resolve_model_profile(QUALITY_PROFILE_KEY)

    @app.function(image=web_image, secrets=[runtime_secret], timeout=900)
    def remote_smoke() -> dict[str, object]:
        from dota2tuned.ui.runtime_hooks import (
            close_idle_main_event_loop,
            install_quiet_unraisablehook,
        )

        install_quiet_unraisablehook()
        from dota2tuned.ui.gradio_app import build_app

        demo = build_app()
        close_idle_main_event_loop()
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

        from dota2tuned.ui.runtime_hooks import (
            close_idle_main_event_loop,
            install_quiet_unraisablehook,
        )

        install_quiet_unraisablehook()
        from dota2tuned.ui.gradio_app import (
            APP_CSS,
            APP_HEAD,
            build_app,
        )

        demo = build_app()
        close_idle_main_event_loop()
        return mount_gradio_app(
            app=FastAPI(),
            blocks=demo,
            path="/",
            css=APP_CSS,
            head=APP_HEAD,
        )

    def _run_train_sft(
        dataset_source: str = str(REMOTE_DATA / "models" / "sft_examples.jsonl"),
        profile: str | None = None,
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
        selected = resolve_model_profile(profile)
        train_settings = Settings(
            hf_token=os.environ.get("HF_TOKEN"),
            model_profile=selected.key,
            hf_model_repo_id=selected.hf_model_repo_id,
            hf_dataset_repo_id=os.environ["HF_DATASET_REPO_ID"],
            base_model_id=selected.base_model_id,
            sft_max_length=selected.sft_max_length,
            lora_r=selected.lora_r,
            lora_alpha=selected.lora_alpha,
            lora_dropout=selected.lora_dropout,
            lora_target_modules=selected.lora_target_modules,
            sft_learning_rate=selected.sft_learning_rate,
            sft_epochs=selected.sft_epochs,
            sft_batch_size=selected.sft_batch_size,
            sft_grad_accum=selected.sft_grad_accum,
            model_load_in_4bit=selected.model_load_in_4bit,
            model_torch_dtype=selected.model_torch_dtype,
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
            "profile": selected.key,
            "base_model": train_settings.base_model_id,
            "output_repo": train_settings.hf_model_repo_id,
            "dataset_source": dataset_source,
            "script": str(script_path),
        }

    @app.function(
        image=train_image,
        gpu=settings.modal_train_gpu,
        secrets=[runtime_secret],
        volumes={REMOTE_CACHE: cache_volume, REMOTE_OUTPUTS: output_volume},
        timeout=settings.modal_train_timeout,
    )
    def train_sft(
        dataset_source: str = str(REMOTE_DATA / "models" / "sft_examples.jsonl"),
        profile: str | None = None,
    ) -> dict[str, object]:
        return _run_train_sft(dataset_source, profile)

    @app.function(
        image=train_image,
        gpu=quality_train_profile.modal_train_gpu,
        secrets=[runtime_secret],
        volumes={REMOTE_CACHE: cache_volume, REMOTE_OUTPUTS: output_volume},
        timeout=quality_train_profile.modal_train_timeout,
    )
    def train_sft_quality(
        dataset_source: str = str(REMOTE_DATA / "models" / "sft_examples.jsonl"),
        profile: str | None = None,
    ) -> dict[str, object]:
        return _run_train_sft(dataset_source, profile or quality_train_profile.key)

    def _run_generate_answer(
        question: str,
        context: str = "",
        max_new_tokens: int = 384,
        profile: str | None = None,
        thinking: bool = False,
    ) -> dict[str, object]:
        import torch
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer, BitsAndBytesConfig

        from dota2tuned.model_profiles import resolve_model_profile

        _install_peft_weight_converter_compat()

        try:
            from transformers.utils import import_utils

            if not hasattr(import_utils, "is_torch_fx_available"):

                def _is_torch_fx_available():
                    try:
                        import torch.fx  # noqa: F401

                        return True
                    except Exception:
                        return False

                import_utils.is_torch_fx_available = _is_torch_fx_available
        except Exception:
            pass

        if not question.strip():
            raise RuntimeError("question is required.")

        selected = resolve_model_profile(profile)
        thinking_enabled = bool(thinking) and bool(selected.supports_thinking)
        model_id = selected.hf_model_repo_id
        cache_key = f"{selected.key}:{model_id}"
        if cache_key not in _MODEL_CACHE:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoPeftModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            model.config.use_cache = False
            model.eval()
            _MODEL_CACHE[cache_key] = {"tokenizer": tokenizer, "model": model}

        cached = _MODEL_CACHE[cache_key]
        tokenizer = cached["tokenizer"]
        model = cached["model"]
        system = (
            "You are DOTA2Tuned, a Dota 2 draft and meta assistant. "
            "Use only the supplied evidence when it contains concrete facts. "
            "Never invent hero names, item names, scores, timings, patches, or match data. "
            "If evidence is missing or too generic, say what is uncertain and ask for more "
            "draft or stat context. Be concise and caveat weak data."
        )
        user = question.strip()
        if context.strip():
            user = f"{user}\n\nEvidence:\n{context.strip()}"
        if selected.key == BALANCED_PROFILE_KEY:
            thinking_tag = "/think" if thinking_enabled else "/no_think"
            messages = [
                {
                    "role": "user",
                    "content": f"{system}\n\n{user}\n\n{thinking_tag}",
                }
            ]
        else:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        try:
            template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if selected.key == BALANCED_PROFILE_KEY:
                template_kwargs["enable_thinking"] = thinking_enabled
            prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
        except Exception:
            prompt = f"{system}\n\nUser: {user}\nAssistant:"
        token_limit = THINKING_TOKEN_LIMIT if thinking_enabled else NON_THINKING_TOKEN_LIMIT
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        generation_kwargs = {
            "max_new_tokens": max(32, min(max_new_tokens, token_limit)),
            "use_cache": False,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if selected.key == BALANCED_PROFILE_KEY:
            generation_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "repetition_penalty": 1.05,
                }
            )
        else:
            generation_kwargs["do_sample"] = False
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **generation_kwargs,
            )
        generated = outputs[0][inputs["input_ids"].shape[-1] :]
        raw_answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
        answer = _strip_reasoning_blocks(raw_answer)
        fallback_reason = None
        if selected.key == BALANCED_PROFILE_KEY and _looks_malformed_answer(answer):
            answer = _evidence_fallback_answer(question, context)
            fallback_reason = "balanced_malformed_generation"
        response = {
            "status": "ok",
            "profile": selected.key,
            "model": model_id,
            "answer": answer,
            "tokens": int(generated.numel()),
            "thinking_requested": bool(thinking),
            "thinking_enabled": thinking_enabled,
            "recommended_max_tokens": (
                selected.thinking_recommended_max_tokens
                if selected.supports_thinking
                else NON_THINKING_TOKEN_LIMIT
            ),
        }
        if fallback_reason:
            response["fallback"] = fallback_reason
        return response

    @app.function(
        image=train_image,
        gpu=settings.modal_infer_gpu,
        secrets=[runtime_secret],
        volumes={REMOTE_CACHE: cache_volume},
        timeout=settings.modal_infer_timeout,
        max_containers=1,
    )
    def generate_answer(
        question: str,
        context: str = "",
        max_new_tokens: int = 384,
        profile: str | None = None,
        thinking: bool = False,
    ) -> dict[str, object]:
        return _run_generate_answer(question, context, max_new_tokens, profile, thinking)

    @app.function(
        image=train_image,
        gpu=quality_train_profile.modal_infer_gpu,
        secrets=[runtime_secret],
        volumes={REMOTE_CACHE: cache_volume},
        timeout=quality_train_profile.modal_infer_timeout,
        max_containers=1,
    )
    def generate_answer_quality(
        question: str,
        context: str = "",
        max_new_tokens: int = 384,
        profile: str | None = None,
        thinking: bool = False,
    ) -> dict[str, object]:
        return _run_generate_answer(
            question,
            context,
            max_new_tokens,
            profile or quality_train_profile.key,
            thinking,
        )
else:
    app = None


def modal_available() -> bool:
    return app is not None
