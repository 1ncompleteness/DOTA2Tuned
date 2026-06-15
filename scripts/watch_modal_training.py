from __future__ import annotations

import argparse
import datetime as dt
import time

import modal


def _status(name: str, call_id: str) -> bool:
    call = modal.FunctionCall.from_id(call_id)
    try:
        result = call.get(timeout=1)
    except TimeoutError:
        print(f"{name} {call_id} running", flush=True)
        return False
    except Exception as exc:
        print(f"{name} {call_id} error {type(exc).__name__}: {str(exc)[:1000]}", flush=True)
        return True
    print(f"{name} {call_id} done {result}", flush=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Poll Modal fine-tuning function calls.")
    parser.add_argument(
        "--call",
        action="append",
        default=[],
        metavar="NAME=CALL_ID",
        help="Modal function call to poll. May be repeated.",
    )
    parser.add_argument("--interval", type=int, default=300, help="Seconds between polls.")
    parser.add_argument(
        "--exit-when-done",
        action="store_true",
        help="Exit after every call is done or failed.",
    )
    args = parser.parse_args()
    calls: dict[str, str] = {}
    for raw in args.call:
        if "=" not in raw:
            raise SystemExit(f"Invalid --call value {raw!r}; expected NAME=CALL_ID.")
        name, call_id = raw.split("=", 1)
        calls[name.strip()] = call_id.strip()
    if not calls:
        raise SystemExit("Provide at least one --call NAME=CALL_ID.")

    while True:
        print(dt.datetime.now(dt.UTC).isoformat(), flush=True)
        done = [_status(name, call_id) for name, call_id in calls.items()]
        all_done = all(done)
        print(f"all_done={all_done}", flush=True)
        if all_done and args.exit_when_done:
            return 0
        time.sleep(max(args.interval, 10))


if __name__ == "__main__":
    raise SystemExit(main())
