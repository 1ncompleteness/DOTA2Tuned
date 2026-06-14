from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from dota2tuned.clients import OpenDotaClient, SteamClient, StratzClient, ValvePatchClient
from dota2tuned.clients.valve import flatten_patch_notes
from dota2tuned.config import Settings


@dataclass(frozen=True)
class SmokeResult:
    name: str
    status: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _has_secret(value: str | None) -> bool:
    return bool(value and value.strip() and not value.endswith("_xxx"))


def _ok(name: str, detail: str) -> SmokeResult:
    return SmokeResult(name=name, status="ok", detail=detail)


def _warn(name: str, detail: str) -> SmokeResult:
    return SmokeResult(name=name, status="warn", detail=detail)


def _fail(name: str, exc: BaseException | str) -> SmokeResult:
    detail = str(exc)
    return SmokeResult(name=name, status="fail", detail=detail[:500])


def _missing(name: str, live: bool) -> SmokeResult:
    if live:
        return SmokeResult(name=name, status="fail", detail="required token is missing")
    return SmokeResult(name=name, status="warn", detail="token not configured")


def check_env(settings: Settings, *, live: bool = False) -> list[SmokeResult]:
    results = [
        _ok("BASE_MODEL_ID", settings.base_model_id),
        _ok("HF_MODEL_REPO_ID", settings.hf_model_repo_id),
        _ok("HF_DATASET_REPO_ID", settings.hf_dataset_repo_id),
        _ok("TRAINING_FLAVOR", settings.training_flavor),
        _ok("HF_JOB_TIMEOUT", settings.hf_job_timeout),
        _ok("SFT_MAX_LENGTH", str(settings.sft_max_length)),
    ]

    for name, value in [
        ("HF_TOKEN", settings.hf_token),
        ("STRATZ_TOKEN", settings.stratz_token),
        ("OPENDOTA_API_KEY", settings.opendota_api_key),
        ("STEAM_API_KEY", settings.steam_api_key),
    ]:
        results.append(_ok(name, "configured") if _has_secret(value) else _missing(name, live=live))
    return results


def _latest_patch_name(rows: Any) -> str:
    if isinstance(rows, dict):
        rows = list(rows.values())
    rows = [row for row in rows or [] if row.get("name")]
    rows.sort(key=lambda row: str(row.get("date", "")))
    return rows[-1]["name"] if rows else "7.41d"


def check_huggingface(settings: Settings) -> SmokeResult:
    if not _has_secret(settings.hf_token):
        return _missing("huggingface", live=True)
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=settings.hf_token)
        user = api.whoami()
        api.model_info(settings.base_model_id)
        name = user.get("name") or user.get("fullname") or "authenticated"
        return _ok("huggingface", f"authenticated as {name}; base model visible")
    except Exception as exc:
        return _fail("huggingface", exc)


def check_opendota(settings: Settings) -> tuple[SmokeResult, str]:
    client = OpenDotaClient(settings.opendota_api_key)
    try:
        metadata = client.metadata()
        patch_name = _latest_patch_name(client.constants("patch"))
        limits = metadata.get("limits") or {}
        limit_text = limits.get("minute") or limits.get("day") or "metadata returned"
        return _ok(
            "opendota", f"metadata ok; latest patch {patch_name}; limit {limit_text}"
        ), patch_name
    except Exception as exc:
        return _fail("opendota", exc), "7.41d"
    finally:
        client.close()


def check_stratz(settings: Settings) -> SmokeResult:
    if not _has_secret(settings.stratz_token):
        return _missing("stratz", live=True)
    client = StratzClient(settings.stratz_token)
    try:
        payload = client.execute("query Smoke { __typename }")
        typename = payload.get("__typename", "query")
        return _ok("stratz", f"GraphQL authenticated; root {typename}")
    except Exception as exc:
        return _fail("stratz", exc)
    finally:
        client.close()


def check_steam(settings: Settings) -> SmokeResult:
    if not _has_secret(settings.steam_api_key):
        return _missing("steam", live=True)
    client = SteamClient(settings.steam_api_key)
    try:
        heroes = client.get_heroes()
        count = len(heroes.get("heroes") or [])
        return _ok("steam", f"GetHeroes returned {count} heroes")
    except Exception as exc:
        return _fail("steam", exc)
    finally:
        client.close()


def check_valve_patch(version: str) -> SmokeResult:
    client = ValvePatchClient()
    try:
        payload = client.patch_notes(version)
        changes = flatten_patch_notes(payload)
        return _ok("valve_patch", f"patch {version} returned {len(changes)} changes")
    except Exception as exc:
        return _fail("valve_patch", exc)
    finally:
        client.close()


def run_smoke_checks(settings: Settings, *, live: bool = False) -> list[dict[str, str]]:
    results = check_env(settings, live=live)
    if live:
        results.append(check_huggingface(settings))
        opendota_result, patch_name = check_opendota(settings)
        results.append(opendota_result)
        results.append(check_stratz(settings))
        results.append(check_steam(settings))
        results.append(check_valve_patch(patch_name))
    return [result.to_dict() for result in results]


def has_failures(results: list[dict[str, str]]) -> bool:
    return any(result["status"] == "fail" for result in results)
