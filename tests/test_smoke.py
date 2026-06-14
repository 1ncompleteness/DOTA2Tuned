from dota2tuned.config import Settings
from dota2tuned.smoke import check_env, has_failures


def test_check_env_does_not_fail_without_live_tokens(tmp_path):
    settings = Settings(
        hf_token=None,
        stratz_token=None,
        opendota_api_key=None,
        steam_api_key=None,
        raw_data_dir=tmp_path / "raw",
        parquet_dir=tmp_path / "parquet",
        rag_dir=tmp_path / "rag",
        model_dir=tmp_path / "models",
        duckdb_path=tmp_path / "db.duckdb",
    )

    results = [result.to_dict() for result in check_env(settings, live=False)]

    assert not has_failures(results)
    assert {result["status"] for result in results} == {"ok", "warn"}


def test_check_env_fails_missing_tokens_for_live_smoke(tmp_path):
    settings = Settings(
        hf_token=None,
        stratz_token=None,
        opendota_api_key=None,
        steam_api_key=None,
        raw_data_dir=tmp_path / "raw",
        parquet_dir=tmp_path / "parquet",
        rag_dir=tmp_path / "rag",
        model_dir=tmp_path / "models",
        duckdb_path=tmp_path / "db.duckdb",
    )

    results = [result.to_dict() for result in check_env(settings, live=True)]

    assert has_failures(results)
