#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

mkdir -p runs

threshold="${DOTA2TUNED_CONFIDENCE_THRESHOLD:-500}"
target="${DOTA2TUNED_CONFIDENCE_TARGET:-12000}"
step="${DOTA2TUNED_CONFIDENCE_STEP:-5000}"
max_target="${DOTA2TUNED_CONFIDENCE_MAX_TARGET:-30000}"
public_matches="${DOTA2TUNED_PUBLIC_MATCHES:-500}"
league_limit="${DOTA2TUNED_LEAGUE_LIMIT:-100}"
stratz_limit="${DOTA2TUNED_STRATZ_LIMIT:-0}"
patch_count="${DOTA2TUNED_PATCH_COUNT:-4}"

raw_details="data/raw/matches/opendota_match_details.jsonl"

raw_count() {
  if [[ -f "$raw_details" ]]; then
    wc -l < "$raw_details"
  else
    echo 0
  fi
}

audit() {
  local stamp
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  uv run python scripts/audit_confidence.py \
    --threshold "$threshold" \
    --output "runs/confidence_audit_${stamp}.json" \
    "$@"
}

echo "confidence expansion started at $(date -u)"
echo "threshold=$threshold target=$target step=$step max_target=$max_target"

if audit --fail-under-max; then
  echo "already at max confidence"
  exit 0
fi

while (( target <= max_target )); do
  before="$(raw_count)"
  if (( target <= before )); then
    target=$((before + step))
  fi
  if (( target > max_target )); then
    break
  fi

  pro_matches="${DOTA2TUNED_PRO_MATCHES:-$((target + 1000))}"
  echo "starting expansion target=$target current_raw=$before pro_matches=$pro_matches"
  uv run dota2tuned ingest \
    --pro-matches "$pro_matches" \
    --public-matches "$public_matches" \
    --enrich-limit "$target" \
    --stratz-limit "$stratz_limit" \
    --patch-count "$patch_count" \
    --league-limit "$league_limit"
  uv run dota2tuned normalize
  uv run dota2tuned features
  uv run dota2tuned train-predictor
  uv run dota2tuned build-rag
  uv run dota2tuned make-sft --limit 600

  after="$(raw_count)"
  echo "finished expansion target=$target raw_before=$before raw_after=$after"
  if audit --fail-under-max; then
    echo "max confidence reached at target=$target"
    exit 0
  fi
  if (( after <= before )); then
    echo "source exhausted or no new match details fetched; stopping at raw_count=$after"
    exit 2
  fi
  target=$((target + step))
done

echo "max target reached before every hero reached high confidence"
audit || true
exit 2
