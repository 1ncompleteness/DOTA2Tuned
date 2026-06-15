from dota2tuned.model_profiles import apply_profile_env_overrides, resolve_model_profile


def test_model_profiles_include_quality_and_sponsor_paths():
    quality = resolve_model_profile("qwen3_30b_a3b_2507")
    sponsor = resolve_model_profile("minicpm4_1_8b")

    assert quality.base_model_id == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert quality.sft_max_length == 4096
    assert quality.modal_train_gpu == "H200"
    assert "q_proj" in quality.lora_target_modules
    assert sponsor.base_model_id == "openbmb/MiniCPM4.1-8B"
    assert sponsor.hf_model_repo_id.endswith("minicpm4-1-8b-lora")


def test_model_profile_env_overrides(monkeypatch):
    monkeypatch.setenv("BASE_MODEL_ID", "example/base")
    monkeypatch.setenv("HF_MODEL_REPO_ID", "example/adapter")
    monkeypatch.setenv("LORA_R", "12")

    profile = apply_profile_env_overrides(resolve_model_profile("qwen3_4b_2507"))

    assert profile.base_model_id == "example/base"
    assert profile.hf_model_repo_id == "example/adapter"
    assert profile.lora_r == 12
