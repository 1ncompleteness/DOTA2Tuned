import inspect

from dota2tuned.modal_backend import (
    _evidence_fallback_answer,
    _install_peft_weight_converter_compat,
    _looks_malformed_answer,
    _strip_reasoning_blocks,
    modal_infer_function_name,
)
from dota2tuned.model_profiles import apply_profile_env_overrides, resolve_model_profile


def test_model_profiles_include_quality_and_sponsor_paths():
    quality = resolve_model_profile("qwen3_30b_a3b_2507")
    sponsor = resolve_model_profile("minicpm4_1_8b")

    assert quality.base_model_id == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert quality.sft_max_length == 4096
    assert quality.modal_train_gpu == "H200"
    assert quality.lora_dropout == 0.0
    assert "q_proj" in quality.lora_target_modules
    assert quality.supports_thinking is False
    assert sponsor.base_model_id == "openbmb/MiniCPM4.1-8B"
    assert sponsor.hf_model_repo_id.endswith("minicpm4-1-8b-lora")
    assert sponsor.supports_thinking is True
    assert sponsor.thinking_recommended_max_tokens == 768


def test_model_profile_env_overrides(monkeypatch):
    monkeypatch.setenv("BASE_MODEL_ID", "example/base")
    monkeypatch.setenv("HF_MODEL_REPO_ID", "example/adapter")
    monkeypatch.setenv("LORA_R", "12")

    profile = apply_profile_env_overrides(resolve_model_profile("qwen3_4b_2507"))

    assert profile.base_model_id == "example/base"
    assert profile.hf_model_repo_id == "example/adapter"
    assert profile.lora_r == 12


def test_model_profile_thinking_env_overrides(monkeypatch):
    monkeypatch.setenv("MODEL_SUPPORTS_THINKING", "1")
    monkeypatch.setenv("THINKING_RECOMMENDED_MAX_TOKENS", "1024")

    profile = apply_profile_env_overrides(resolve_model_profile("qwen3_4b_2507"))

    assert profile.supports_thinking is True
    assert profile.thinking_recommended_max_tokens == 1024


def test_modal_inference_routes_quality_to_h200_function():
    assert modal_infer_function_name("qwen3_4b_2507") == "generate_answer"
    assert modal_infer_function_name("minicpm4_1_8b") == "generate_answer"
    assert modal_infer_function_name("qwen3_30b_a3b_2507") == "generate_answer_quality"


def test_malformed_balanced_output_guard_returns_grounded_text():
    assert _looks_malformed_answer('".  \\  \\  \\  \\  \\  \\  \\')
    assert _looks_malformed_answer("c3 ×ontology**\n\ninter. ##")
    assert _looks_malformed_answer(
        "vestig vestig vestig vestib vestige vestig vestment vestig vestig"
    )
    assert not _looks_malformed_answer("Crystal Maiden is a support with control.")

    fallback = _evidence_fallback_answer(
        "Suggest one support against Phantom Assassin.",
        "Crystal Maiden has control. Phantom Assassin is a carry.",
    )

    assert "Based on the retrieved evidence" in fallback
    assert "Crystal Maiden has control" in fallback


def test_reasoning_blocks_are_stripped_from_visible_answer():
    assert _strip_reasoning_blocks("<think>private scratchpad</think>Final answer.") == (
        "Final answer."
    )
    assert _strip_reasoning_blocks("private scratchpad</think>Final answer.") == (
        "Final answer."
    )


def test_peft_weight_converter_compat_accepts_new_peft_kwargs():
    installed = _install_peft_weight_converter_compat()
    if not installed:
        return

    from transformers.core_model_loading import WeightConverter

    signature = inspect.signature(WeightConverter.__init__)
    assert "distributed_operation" in signature.parameters
    assert "quantization_operation" in signature.parameters
