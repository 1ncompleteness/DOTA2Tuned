import pytest

from dota2tuned.config import Settings
from dota2tuned.finetune import _token_permissions, validate_hf_jobs_access


def test_token_permissions_use_requested_namespace_only():
    whoami = {
        "name": "alice",
        "auth": {
            "accessToken": {
                "role": "fineGrained",
                "fineGrained": {
                    "global": [],
                    "scoped": [
                        {
                            "entity": {"type": "user", "name": "alice"},
                            "permissions": ["job.write"],
                        },
                        {
                            "entity": {"type": "org", "name": "team"},
                            "permissions": ["repo.write"],
                        },
                    ],
                },
            }
        },
    }

    assert _token_permissions(whoami, "team") == {"repo.write"}


def test_validate_hf_jobs_access_requires_job_write(monkeypatch, tmp_path):
    class FakeApi:
        def __init__(self, token):
            self.token = token

        def whoami(self):
            return {
                "name": "alice",
                "auth": {
                    "accessToken": {
                        "role": "fineGrained",
                        "fineGrained": {
                            "global": [],
                            "scoped": [
                                {
                                    "entity": {"type": "org", "name": "team"},
                                    "permissions": ["repo.write"],
                                }
                            ],
                        },
                    }
                },
            }

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    settings = Settings(
        hf_token="hf_test",
        hf_org="team",
        training_flavor="a100-large",
        raw_data_dir=tmp_path / "raw",
        parquet_dir=tmp_path / "parquet",
        rag_dir=tmp_path / "rag",
        model_dir=tmp_path / "models",
        duckdb_path=tmp_path / "db.duckdb",
    )

    with pytest.raises(RuntimeError, match="job.write"):
        validate_hf_jobs_access(settings)
