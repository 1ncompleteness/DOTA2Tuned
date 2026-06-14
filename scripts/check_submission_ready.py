from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from check_public_space import DEFAULT_SPACE, run_checks
from huggingface_hub import HfApi

DEFAULT_MODEL_REPO = "build-small-hackathon/dota2tuned-qwen3-4b-2507-lora"
DEFAULT_DATASET_REPO = "build-small-hackathon/dota2tuned-data"
DEFAULT_MODAL_URL = "https://dracufeuer--dota2tuned-ui.modal.run"
DEFAULT_GITHUB_REPO = "https://github.com/1ncompleteness/DOTA2Tuned"

MANUAL_ITEMS = [
    "Record short demo video.",
    "Publish social post.",
    "Submit Space link, demo video link, and social post link by June 15, 2026.",
]


def _http_status(url: str, *, timeout: int = 30) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": "DOTA2Tuned submission checker"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return {"ok": 200 <= response.status < 400, "status": response.status}
    except HTTPError as exc:
        return {"ok": False, "status": exc.code, "error": str(exc)}
    except URLError as exc:
        return {"ok": False, "status": None, "error": str(exc.reason)}


def _git_status() -> dict[str, Any]:
    result = subprocess.run(
        ["git", "status", "--short", "--branch"],
        check=False,
        capture_output=True,
        text=True,
    )
    lines = result.stdout.strip().splitlines()
    changed = [line for line in lines[1:] if line.strip()]
    return {
        "ok": result.returncode == 0 and not changed,
        "returncode": result.returncode,
        "status": lines,
    }


def _repo_has_file(api: HfApi, repo_id: str, repo_type: str, filename: str) -> dict[str, Any]:
    try:
        if repo_type == "model":
            info = api.model_info(repo_id)
        elif repo_type == "dataset":
            info = api.dataset_info(repo_id)
        else:
            raise ValueError(f"unsupported repo_type: {repo_type}")
        files = {sibling.rfilename for sibling in info.siblings}
        return {"ok": filename in files, "file": filename, "repo": repo_id}
    except Exception as exc:  # pragma: no cover - network smoke script
        return {"ok": False, "file": filename, "repo": repo_id, "error": repr(exc)}


def _space_runtime(api: HfApi, space_id: str) -> dict[str, Any]:
    try:
        info = api.space_info(space_id)
        runtime = getattr(info, "runtime", None)
        stage = getattr(runtime, "stage", str(runtime))
        return {"ok": stage == "RUNNING", "stage": stage, "space": space_id}
    except Exception as exc:  # pragma: no cover - network smoke script
        return {"ok": False, "space": space_id, "error": repr(exc)}


def run_submission_checks(args: argparse.Namespace) -> dict[str, Any]:
    token = os.getenv("HF_TOKEN")
    api = HfApi(token=token)
    space_id = args.space_repo
    checks: dict[str, Any] = {
        "git_clean": _git_status(),
        "space_http": _http_status(args.space_url),
        "modal_http": _http_status(args.modal_url),
        "github_http": _http_status(args.github_repo),
        "space_runtime": _space_runtime(api, space_id),
        "model_card": _repo_has_file(api, args.model_repo, "model", "README.md"),
        "dataset_card": _repo_has_file(api, args.dataset_repo, "dataset", "README.md"),
    }
    if not args.skip_public_api:
        public_api = run_checks(args.space_url)
        checks["public_space_api"] = {
            "ok": all(item["status"] == "ok" for item in public_api["checks"]),
            "checks": public_api["checks"],
        }
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "machine_checks": checks,
        "manual_remaining": MANUAL_ITEMS,
        "links": {
            "space": args.space_url,
            "github": args.github_repo,
            "model": f"https://huggingface.co/{args.model_repo}",
            "dataset": f"https://huggingface.co/datasets/{args.dataset_repo}",
            "modal": args.modal_url,
        },
    }


def _machine_ok(payload: dict[str, Any]) -> bool:
    return all(check.get("ok") is True for check in payload["machine_checks"].values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final DOTA2Tuned submission checks.")
    parser.add_argument("--space-url", default=DEFAULT_SPACE)
    parser.add_argument("--space-repo", default="build-small-hackathon/dota2tuned")
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--dataset-repo", default=DEFAULT_DATASET_REPO)
    parser.add_argument("--modal-url", default=DEFAULT_MODAL_URL)
    parser.add_argument("--github-repo", default=DEFAULT_GITHUB_REPO)
    parser.add_argument(
        "--output",
        default="submission_evidence/final_submission_check.json",
        help="Path for JSON evidence output.",
    )
    parser.add_argument(
        "--skip-public-api",
        action="store_true",
        help="Skip Gradio endpoint checks when only public artifact status is needed.",
    )
    args = parser.parse_args()

    payload = run_submission_checks(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Machine checks:")
    for name, check in payload["machine_checks"].items():
        detail = check.get("stage") or check.get("status") or check.get("file") or ""
        print(f"- {name}: {'ok' if check.get('ok') else 'fail'} {detail}")
    print("\nManual remaining:")
    for item in payload["manual_remaining"]:
        print(f"- {item}")
    print(f"\nWrote {output_path}")

    if not _machine_ok(payload):
        sys.exit(1)


if __name__ == "__main__":
    main()
