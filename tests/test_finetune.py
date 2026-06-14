import pytest

from dota2tuned.config import Settings
from dota2tuned.finetune import _token_permissions, launch_hf_job, validate_hf_jobs_access


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
        hf_jobs_token="hf_jobs",
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


def test_validate_hf_jobs_access_allows_split_tokens(monkeypatch, tmp_path):
    class Hardware:
        name = "a100-large"

    class FakeApi:
        def __init__(self, token):
            self.token = token

        def whoami(self):
            permissions = ["job.write"] if self.token == "hf_jobs" else ["repo.write"]
            return {
                "name": self.token,
                "auth": {
                    "accessToken": {
                        "role": "fineGrained",
                        "fineGrained": {
                            "global": [],
                            "scoped": [
                                {
                                    "entity": {"type": "org", "name": "team"},
                                    "permissions": permissions,
                                }
                            ],
                        },
                    }
                },
            }

        def list_jobs_hardware(self):
            return [Hardware()]

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    settings = Settings(
        hf_token="hf_repo",
        hf_jobs_token="hf_jobs",
        hf_org="team",
        training_flavor="a100-large",
        raw_data_dir=tmp_path / "raw",
        parquet_dir=tmp_path / "parquet",
        rag_dir=tmp_path / "rag",
        model_dir=tmp_path / "models",
        duckdb_path=tmp_path / "db.duckdb",
    )

    assert validate_hf_jobs_access(settings) == {
        "namespace": "team",
        "flavor": "a100-large",
        "user": "hf_jobs",
    }


def test_validate_hf_jobs_access_requires_repo_write(monkeypatch, tmp_path):
    class FakeApi:
        def __init__(self, token):
            self.token = token

        def whoami(self):
            permissions = ["job.write"] if self.token == "hf_jobs" else ["repo.content.read"]
            return {
                "name": self.token,
                "auth": {
                    "accessToken": {
                        "role": "fineGrained",
                        "fineGrained": {
                            "global": [],
                            "scoped": [
                                {
                                    "entity": {"type": "org", "name": "team"},
                                    "permissions": permissions,
                                }
                            ],
                        },
                    }
                },
            }

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    settings = Settings(
        hf_token="hf_repo",
        hf_jobs_token="hf_jobs",
        hf_org="team",
        training_flavor="a100-large",
        raw_data_dir=tmp_path / "raw",
        parquet_dir=tmp_path / "parquet",
        rag_dir=tmp_path / "rag",
        model_dir=tmp_path / "models",
        duckdb_path=tmp_path / "db.duckdb",
    )

    with pytest.raises(RuntimeError, match="repo.write"):
        validate_hf_jobs_access(settings)


def test_launch_hf_job_uses_jobs_token(monkeypatch, tmp_path):
    calls = {}

    class FakeApi:
        def __init__(self, token):
            calls["create_repo_token"] = token

        def create_repo(self, **kwargs):
            calls["create_repo"] = kwargs

    def fake_run_uv_job(script, **kwargs):
        calls["script"] = script
        calls["run_uv_job"] = kwargs
        return "job-123"

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    monkeypatch.setattr("huggingface_hub.run_uv_job", fake_run_uv_job)
    script_path = tmp_path / "train.py"
    script_path.write_text("print('train')\n")
    settings = Settings(
        hf_token="hf_repo",
        hf_jobs_token="hf_jobs",
        hf_org="team",
        hf_model_repo_id="team/model",
        training_flavor="a100-large",
        hf_job_timeout="6h",
        raw_data_dir=tmp_path / "raw",
        parquet_dir=tmp_path / "parquet",
        rag_dir=tmp_path / "rag",
        model_dir=tmp_path / "models",
        duckdb_path=tmp_path / "db.duckdb",
    )

    assert launch_hf_job(settings, script_path) == "job-123"
    assert calls["create_repo_token"] == "hf_repo"
    assert calls["create_repo"]["repo_id"] == "team/model"
    assert calls["run_uv_job"]["token"] == "hf_jobs"
    assert calls["run_uv_job"]["secrets"]["HF_TOKEN"] == "hf_repo"
