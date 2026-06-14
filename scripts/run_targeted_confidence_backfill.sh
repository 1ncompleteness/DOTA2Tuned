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
hero_limit="${DOTA2TUNED_TARGETED_HERO_LIMIT:-64}"
matches_per_hero="${DOTA2TUNED_TARGETED_MATCHES_PER_HERO:-700}"
batch_new_details="${DOTA2TUNED_TARGETED_BATCH_NEW_DETAILS:-5000}"
max_rounds="${DOTA2TUNED_TARGETED_MAX_ROUNDS:-8}"
page_size="${DOTA2TUNED_TARGETED_PAGE_SIZE:-100}"
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
    --output "runs/targeted_confidence_audit_${stamp}.json" \
    "$@"
}

rebuild_artifacts() {
  uv run dota2tuned normalize
  uv run dota2tuned features
  uv run dota2tuned train-predictor
  uv run dota2tuned build-rag
  uv run dota2tuned make-sft --limit 600
}

echo "targeted confidence backfill started at $(date -u)"
echo "threshold=$threshold hero_limit=$hero_limit matches_per_hero=$matches_per_hero"
echo "batch_new_details=$batch_new_details max_rounds=$max_rounds page_size=$page_size"

if audit --fail-under-max; then
  echo "already at max confidence"
  exit 0
fi

for round in $(seq 1 "$max_rounds"); do
  before="$(raw_count)"
  echo "round=$round raw_before=$before"
  uv run dota2tuned targeted-ingest \
    --threshold "$threshold" \
    --hero-limit "$hero_limit" \
    --matches-per-hero "$matches_per_hero" \
    --max-new-details "$batch_new_details" \
    --page-size "$page_size"
  after_fetch="$(raw_count)"
  echo "round=$round raw_after_fetch=$after_fetch"
  if (( after_fetch <= before )); then
    echo "no new targeted match details fetched; stopping"
    audit || true
    exit 2
  fi

  rebuild_artifacts
  if audit --fail-under-max; then
    echo "max confidence reached after round=$round"
    exit 0
  fi
done

echo "targeted max rounds reached before every hero reached high confidence"
audit || true
exit 2
