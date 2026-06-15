from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from dota2tuned.clients import OpenDotaClient, StratzClient, ValvePatchClient
from dota2tuned.clients.valve import flatten_patch_notes
from dota2tuned.config import Settings
from dota2tuned.storage import read_jsonl, read_parquet, write_jsonl

PATCH_NOTE_SUFFIXES = ("", "a", "b", "c", "d", "e")


def _take_unique_matches(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    seen: set[int] = set()
    output = []
    for row in rows:
        match_id = row.get("match_id")
        if not match_id or match_id in seen:
            continue
        seen.add(match_id)
        output.append(row)
        if len(output) >= limit:
            break
    return output


def _recent_league_ids(rows: list[dict[str, Any]], limit: int) -> list[int]:
    seen: set[int] = set()
    league_ids: list[int] = []
    for row in rows:
        league_id = row.get("leagueid") or row.get("league_id")
        if not league_id:
            continue
        league_id = int(league_id)
        if league_id in seen:
            continue
        seen.add(league_id)
        league_ids.append(league_id)
        if len(league_ids) >= limit:
            break
    return league_ids


def _patch_note_versions(version: str) -> list[str]:
    base = str(version or "").strip()
    if not base:
        return []
    if base[-1:].isalpha():
        return [base]
    return [f"{base}{suffix}" for suffix in PATCH_NOTE_SUFFIXES]


def _indexed_match_details(path: Path, id_key: str = "match_id") -> dict[int, dict[str, Any]]:
    details = {}
    for row in read_jsonl(path):
        match_id = row.get(id_key) or row.get("id")
        if not match_id:
            continue
        details[int(match_id)] = row
    return details


def _existing_match_ids(path: Path, id_key: str = "match_id") -> set[int]:
    if not path.exists():
        return set()
    ids = set()
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            match_id = row.get(id_key) or row.get("id")
            if match_id:
                ids.add(int(match_id))
    return ids


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    return len(rows)


def _targeted_match_query(hero_id: int, *, limit: int, offset: int) -> str:
    return f"""
select match_id,
  m.start_time,
  m.duration,
  m.radiant_win,
  m.game_mode,
  m.lobby_type,
  m.avg_rank_tier
from public_matches m
where ({hero_id} = any(m.radiant_team) or {hero_id} = any(m.dire_team))
  and m.duration >= 600
  and m.radiant_win is not null
order by m.start_time desc
limit {limit}
offset {offset}
""".strip()


def _under_sampled_heroes(
    parquet_dir: Path, *, threshold: int, limit: int
) -> list[dict[str, int | str]]:
    heroes = read_parquet(parquet_dir / "dim_hero.parquet")
    players = read_parquet(parquet_dir / "fact_player_match.parquet")
    if heroes.is_empty() or players.is_empty():
        return []
    if "pro_pick" not in heroes.columns:
        heroes = heroes.with_columns(pl.lit(0).alias("pro_pick"))
    counts = players.group_by("hero_id").agg(pl.len().cast(pl.Int64).alias("player_games"))
    gaps = (
        heroes.select(["hero_id", "hero_name", "pro_pick"])
        .join(counts, on="hero_id", how="left")
        .with_columns(pl.col("player_games").fill_null(0))
        .with_columns(pl.max_horizontal("player_games", "pro_pick").alias("sample_size"))
        .filter(pl.col("sample_size") < threshold)
        .with_columns((pl.lit(threshold) - pl.col("sample_size")).alias("needed"))
        .sort("needed", descending=True)
        .head(limit)
    )
    return [
        {
            "hero_id": int(row["hero_id"]),
            "hero_name": str(row["hero_name"]),
            "sample_size": int(row["sample_size"]),
            "needed": int(row["needed"]),
        }
        for row in gaps.iter_rows(named=True)
    ]


class IngestCoordinator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.raw_dir = settings.raw_data_dir

    def ingest_reference(self) -> dict[str, int]:
        client = OpenDotaClient(self.settings.opendota_api_key)
        try:
            counts = {
                "opendota_metadata": write_jsonl(
                    self.raw_dir / "reference" / "opendota_metadata.jsonl", [client.metadata()]
                ),
                "opendota_hero_stats": write_jsonl(
                    self.raw_dir / "reference" / "opendota_hero_stats.jsonl",
                    client.hero_stats(),
                ),
            }
            for resource in ["heroes", "items", "patch", "game_mode", "lobby_type", "region"]:
                rows = client.constants(resource)
                if isinstance(rows, dict):
                    rows = [
                        {"key": key, **value}
                        if isinstance(value, dict)
                        else {"key": key, "value": value}
                        for key, value in rows.items()
                    ]
                counts[f"constants_{resource}"] = write_jsonl(
                    self.raw_dir / "reference" / f"opendota_constants_{resource}.jsonl", rows
                )
            counts["opendota_leagues"] = write_jsonl(
                self.raw_dir / "reference" / "opendota_leagues.jsonl", client.leagues()
            )
            return counts
        finally:
            client.close()

    def ingest_patches(self, max_patches: int = 4) -> dict[str, int]:
        od = OpenDotaClient(self.settings.opendota_api_key)
        valve = ValvePatchClient()
        try:
            patch_rows = od.constants("patch")
            if isinstance(patch_rows, dict):
                patch_rows = list(patch_rows.values())
            patch_rows = sorted(patch_rows, key=lambda row: row.get("date", ""))[-max_patches:]
            changes: list[dict[str, Any]] = []
            seen_versions: set[str] = set()
            for row in patch_rows:
                version = row.get("name")
                if not version:
                    continue
                for note_version in _patch_note_versions(str(version)):
                    try:
                        payload = valve.patch_notes(note_version)
                    except Exception:
                        continue
                    rows = flatten_patch_notes(payload)
                    if not rows:
                        continue
                    actual_version = str(rows[0].get("patch") or note_version)
                    if actual_version in seen_versions:
                        continue
                    seen_versions.add(actual_version)
                    changes.extend(rows)
            return {
                "valve_patch_changes": write_jsonl(
                    self.raw_dir / "patches" / "valve_patch_changes.jsonl", changes
                )
            }
        finally:
            od.close()
            valve.close()

    def ingest_matches(
        self,
        *,
        pro_matches: int,
        public_matches: int,
        enrich_limit: int,
        stratz_limit: int = 0,
        league_limit: int = 3,
    ) -> dict[str, int]:
        od = OpenDotaClient(self.settings.opendota_api_key)
        stratz = StratzClient(self.settings.stratz_token) if self.settings.stratz_token else None
        try:
            latest_pro = od.pro_matches()
            league_rows: list[dict[str, Any]] = []
            for league_id in _recent_league_ids(latest_pro, league_limit):
                try:
                    league_rows.extend(od.league_matches(league_id))
                except Exception as exc:
                    league_rows.append(
                        {
                            "match_id": None,
                            "leagueid": league_id,
                            "error": str(exc),
                            "source": "opendota",
                        }
                    )
            pro = _take_unique_matches(latest_pro + league_rows, pro_matches)
            public = [
                row
                for row in _take_unique_matches(od.public_matches(), public_matches * 2)
                if row.get("duration", 0) >= 600 and row.get("radiant_win") is not None
            ][:public_matches]
            selected = _take_unique_matches(pro + public, enrich_limit)
            details_path = self.raw_dir / "matches" / "opendota_match_details.jsonl"
            stratz_path = self.raw_dir / "matches" / "stratz_match_details.jsonl"
            detail_by_match = _indexed_match_details(details_path)
            stratz_by_match = _indexed_match_details(stratz_path, id_key="id")
            for row in selected:
                match_id = int(row["match_id"])
                if match_id not in detail_by_match:
                    try:
                        detail_by_match[match_id] = od.match(match_id)
                    except Exception as exc:
                        detail_by_match[match_id] = {
                            "match_id": match_id,
                            "error": str(exc),
                            "source": "opendota",
                        }
                    if len(detail_by_match) % 25 == 0:
                        write_jsonl(details_path, detail_by_match.values())
                if (
                    stratz
                    and len(stratz_by_match) < stratz_limit
                    and match_id not in stratz_by_match
                ):
                    try:
                        stratz_by_match[match_id] = stratz.match(match_id)
                    except Exception as exc:
                        stratz_by_match[match_id] = {
                            "id": match_id,
                            "error": str(exc),
                            "source": "stratz",
                        }
                    if len(stratz_by_match) % 25 == 0:
                        write_jsonl(stratz_path, stratz_by_match.values())

            counts = {
                "opendota_pro_matches": write_jsonl(
                    self.raw_dir / "matches" / "opendota_pro_matches.jsonl", pro
                ),
                "opendota_public_matches": write_jsonl(
                    self.raw_dir / "matches" / "opendota_public_matches.jsonl", public
                ),
                "opendota_league_matches": write_jsonl(
                    self.raw_dir / "matches" / "opendota_league_matches.jsonl",
                    [row for row in league_rows if row.get("match_id")],
                ),
                "opendota_match_details": write_jsonl(
                    details_path,
                    detail_by_match.values(),
                ),
            }
            if stratz_by_match:
                counts["stratz_match_details"] = write_jsonl(stratz_path, stratz_by_match.values())
            return counts
        finally:
            od.close()
            if stratz:
                stratz.close()

    def ingest_targeted_hero_matches(
        self,
        *,
        threshold: int = 500,
        hero_limit: int = 64,
        matches_per_hero: int = 700,
        max_new_details: int = 5000,
        page_size: int = 100,
        checkpoint_every: int = 25,
    ) -> dict[str, Any]:
        od = OpenDotaClient(self.settings.opendota_api_key)
        details_path = self.raw_dir / "matches" / "opendota_match_details.jsonl"
        targeted_path = self.raw_dir / "matches" / "opendota_targeted_matches.jsonl"
        error_path = self.raw_dir / "matches" / "opendota_targeted_errors.jsonl"
        try:
            gaps = _under_sampled_heroes(
                self.settings.parquet_dir, threshold=threshold, limit=hero_limit
            )
            existing_details = _existing_match_ids(details_path)
            targeted_by_match = _indexed_match_details(targeted_path)
            error_ids = _existing_match_ids(error_path)
            fetched = 0
            discovered = 0
            errors: list[dict[str, Any]] = []
            per_hero: list[dict[str, Any]] = []

            for gap in gaps:
                if fetched >= max_new_details:
                    break
                hero_id = int(gap["hero_id"])
                hero_goal = min(matches_per_hero, max(int(gap["needed"]) + 50, 100))
                offset = 0
                hero_discovered = 0
                hero_fetched = 0
                while hero_fetched < hero_goal and fetched < max_new_details:
                    rows = od.explorer(
                        _targeted_match_query(hero_id, limit=page_size, offset=offset)
                    ).get("rows", [])
                    if not rows:
                        break
                    offset += page_size
                    discovered += len(rows)
                    hero_discovered += len(rows)
                    for row in rows:
                        match_id = int(row["match_id"])
                        targeted_by_match.setdefault(
                            match_id,
                                {
                                    "match_id": match_id,
                                    "hero_id": hero_id,
                                    "start_time": row.get("start_time"),
                                    "duration": row.get("duration"),
                                    "radiant_win": row.get("radiant_win"),
                                    "game_mode": row.get("game_mode"),
                                    "lobby_type": row.get("lobby_type"),
                                    "avg_rank_tier": row.get("avg_rank_tier"),
                                "source": "opendota_explorer_targeted",
                            },
                        )
                        if match_id in existing_details or match_id in error_ids:
                            continue
                        try:
                            detail = od.match(match_id)
                        except Exception as exc:
                            errors.append(
                                {
                                    "match_id": match_id,
                                    "hero_id": hero_id,
                                    "error": str(exc),
                                    "source": "opendota_explorer_targeted",
                                }
                            )
                            error_ids.add(match_id)
                            continue
                        _append_jsonl(details_path, [detail])
                        existing_details.add(match_id)
                        fetched += 1
                        hero_fetched += 1
                        if fetched % checkpoint_every == 0:
                            write_jsonl(targeted_path, targeted_by_match.values())
                            if errors:
                                _append_jsonl(error_path, errors)
                                errors = []
                        if hero_fetched >= hero_goal or fetched >= max_new_details:
                            break
                per_hero.append(
                    {
                        **gap,
                        "discovered": hero_discovered,
                        "fetched": hero_fetched,
                    }
                )

            write_jsonl(targeted_path, targeted_by_match.values())
            if errors:
                _append_jsonl(error_path, errors)
            return {
                "targeted_heroes": len(gaps),
                "targeted_discovered_rows": discovered,
                "targeted_new_details": fetched,
                "opendota_match_details": len(existing_details),
                "opendota_targeted_matches": len(targeted_by_match),
                "per_hero": per_hero,
            }
        finally:
            od.close()


def raw_path(settings: Settings, *parts: str) -> Path:
    return settings.raw_data_dir.joinpath(*parts)
