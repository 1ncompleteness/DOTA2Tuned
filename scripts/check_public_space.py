from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gradio_client import Client

DEFAULT_SPACE = "https://build-small-hackathon-dota2tuned.hf.space"


CHECKS: list[tuple[str, tuple[Any, ...]]] = [
    ("/draft_coach", ([], [44, 30], [], "mid", "pro")),
    ("/hero_meta", ("current pro meta", 44, "bfury")),
    (
        "/match_predictor",
        ([1, 2, 3, 25, 5], [14, 74, 6, 26, 18]),
    ),
    ("/hero_builds", (1, None)),
    ("/draft_lab", ([44, 30], "mid", "Tiny scout card")),
    ("/data_status", ()),
    (
        "/tuned_model",
        (
            "Suggest one mid hero against Phantom Assassin and Witch Doctor, "
            "and include one caveat.",
            "",
            160,
        ),
    ),
]


def run_checks(space_url: str) -> dict[str, Any]:
    client = Client(space_url)
    results = []
    for endpoint, args in CHECKS:
        try:
            result = client.predict(*args, api_name=endpoint)
            status = "ok"
        except Exception as exc:  # pragma: no cover - network smoke script
            result = repr(exc)
            status = "error"
        results.append(
            {
                "endpoint": endpoint,
                "status": status,
                "args": args,
                "result": result,
            }
        )
        print(f"{endpoint}: {status}")
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "space": space_url,
        "checks": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run public DOTA2Tuned Space API checks.")
    parser.add_argument("--space", default=DEFAULT_SPACE)
    parser.add_argument(
        "--output",
        default="submission_evidence/public_space_api_checks.json",
        help="Path for JSON evidence output.",
    )
    args = parser.parse_args()

    payload = run_checks(args.space)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
