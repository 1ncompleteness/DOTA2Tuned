from dota2tuned.config import get_settings


def test_settings_paths_are_created():
    settings = get_settings()
    assert settings.raw_data_dir.exists()
    assert settings.parquet_dir.exists()
