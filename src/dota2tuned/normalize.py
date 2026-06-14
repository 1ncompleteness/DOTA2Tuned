from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import polars as pl

from dota2tuned.storage import read_jsonl, write_parquet

SCHEMAS: dict[str, dict[str, pl.DataType]] = {
    "dim_hero": {
        "hero_id": pl.Int64,
        "hero_name": pl.Utf8,
        "primary_attr": pl.Utf8,
        "attack_type": pl.Utf8,
        "roles": pl.Utf8,
        "pro_pick": pl.Int64,
        "pro_win": pl.Int64,
        "pro_win_rate": pl.Float64,
        "img": pl.Utf8,
        "icon": pl.Utf8,
    },
    "dim_item": {
        "item_id": pl.Int64,
        "item_key": pl.Utf8,
        "item_name": pl.Utf8,
        "cost": pl.Int64,
        "secret_shop": pl.Boolean,
        "side_shop": pl.Boolean,
        "recipe": pl.Boolean,
    },
    "dim_patch": {
        "patch_id": pl.Int64,
        "patch_name": pl.Utf8,
        "date": pl.Utf8,
    },
    "dim_league": {
        "leagueid": pl.Int64,
        "name": pl.Utf8,
        "tier": pl.Utf8,
        "region": pl.Utf8,
    },
    "fact_match": {
        "match_id": pl.Int64,
        "match_seq_num": pl.Int64,
        "source": pl.Utf8,
        "radiant_win": pl.Boolean,
        "start_time": pl.Int64,
        "duration": pl.Int64,
        "leagueid": pl.Int64,
        "league_name": pl.Utf8,
        "radiant_team_id": pl.Int64,
        "dire_team_id": pl.Int64,
        "radiant_name": pl.Utf8,
        "dire_name": pl.Utf8,
        "avg_rank_tier": pl.Int64,
        "game_mode": pl.Int64,
        "lobby_type": pl.Int64,
        "valid_for_training": pl.Boolean,
    },
    "fact_player_match": {
        "match_id": pl.Int64,
        "account_id": pl.Int64,
        "player_slot": pl.Int64,
        "is_radiant": pl.Boolean,
        "hero_id": pl.Int64,
        "lane": pl.Int64,
        "lane_role": pl.Int64,
        "rank_tier": pl.Int64,
        "kills": pl.Int64,
        "deaths": pl.Int64,
        "assists": pl.Int64,
        "gold_per_min": pl.Int64,
        "xp_per_min": pl.Int64,
        "win": pl.Int64,
        "patch": pl.Int64,
    },
    "fact_draft_pickban": {
        "match_id": pl.Int64,
        "is_pick": pl.Boolean,
        "hero_id": pl.Int64,
        "team": pl.Int64,
        "order": pl.Int64,
    },
    "fact_item_purchase": {
        "match_id": pl.Int64,
        "account_id": pl.Int64,
        "is_radiant": pl.Boolean,
        "hero_id": pl.Int64,
        "item_key": pl.Utf8,
        "time": pl.Int64,
        "patch": pl.Int64,
    },
    "fact_hero_pair_stats": {
        "hero_id": pl.Int64,
        "other_hero_id": pl.Int64,
        "relation": pl.Utf8,
        "games": pl.Int64,
        "wins": pl.Int64,
        "win_rate": pl.Float64,
    },
    "fact_hero_build_stats": {
        "hero_id": pl.Int64,
        "item_key": pl.Utf8,
        "time_bucket": pl.Utf8,
        "purchases": pl.Int64,
        "median_time": pl.Float64,
    },
    "doc_patch_change": {
        "patch": pl.Utf8,
        "patch_timestamp": pl.Int64,
        "section": pl.Utf8,
        "subject_type": pl.Utf8,
        "subject_id": pl.Int64,
        "text": pl.Utf8,
        "indent_level": pl.Int64,
        "aghanims": pl.Utf8,
        "icon": pl.Utf8,
    },
}


def normalize_hero_stats(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        pro_pick = int(row.get("pro_pick") or 0)
        pro_win = int(row.get("pro_win") or 0)
        output.append(
            {
                "hero_id": int(row["id"]),
                "hero_name": row.get("localized_name") or row.get("name"),
                "primary_attr": row.get("primary_attr"),
                "attack_type": row.get("attack_type"),
                "roles": ",".join(row.get("roles") or []),
                "pro_pick": pro_pick,
                "pro_win": pro_win,
                "pro_win_rate": pro_win / pro_pick if pro_pick else None,
                "img": row.get("img"),
                "icon": row.get("icon"),
            }
        )
    return output


def normalize_items(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        raw_id = row.get("id") or row.get("item_id") or row.get("key")
        try:
            item_id = int(raw_id)
        except (TypeError, ValueError):
            item_id = None
        output.append(
            {
                "item_id": item_id,
                "item_key": str(row.get("key") or row.get("name") or raw_id or ""),
                "item_name": row.get("dname") or row.get("localized_name") or row.get("name"),
                "cost": row.get("cost"),
                "secret_shop": row.get("secret_shop"),
                "side_shop": row.get("side_shop"),
                "recipe": row.get("recipe"),
            }
        )
    return output


def normalize_patches(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        raw_id = row.get("id") or row.get("key")
        try:
            patch_id = int(raw_id)
        except (TypeError, ValueError):
            patch_id = None
        output.append(
            {
                "patch_id": patch_id,
                "patch_name": row.get("name") or str(raw_id or ""),
                "date": str(row.get("date")) if row.get("date") is not None else None,
            }
        )
    return output


def normalize_leagues(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        leagueid = row.get("leagueid") or row.get("league_id") or row.get("id")
        try:
            leagueid = int(leagueid)
        except (TypeError, ValueError):
            continue
        output.append(
            {
                "leagueid": leagueid,
                "name": row.get("name") or row.get("display_name"),
                "tier": row.get("tier"),
                "region": row.get("region"),
            }
        )
    return output


def normalize_match_summaries(rows: list[dict[str, Any]], *, source: str) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        duration = int(row.get("duration") or 0)
        output.append(
            {
                "match_id": int(row["match_id"]),
                "match_seq_num": row.get("match_seq_num"),
                "source": source,
                "radiant_win": row.get("radiant_win"),
                "start_time": row.get("start_time"),
                "duration": duration,
                "leagueid": row.get("leagueid"),
                "league_name": row.get("league_name"),
                "radiant_team_id": row.get("radiant_team_id"),
                "dire_team_id": row.get("dire_team_id"),
                "radiant_name": row.get("radiant_name") or row.get("radiant_team_name"),
                "dire_name": row.get("dire_name") or row.get("dire_team_name"),
                "avg_rank_tier": row.get("avg_rank_tier"),
                "game_mode": row.get("game_mode"),
                "lobby_type": row.get("lobby_type"),
                "valid_for_training": duration >= 600 and row.get("radiant_win") is not None,
            }
        )
    return output


def dedupe_match_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    output = []
    for row in rows:
        match_id = row.get("match_id")
        if not match_id or match_id in seen:
            continue
        seen.add(match_id)
        output.append(row)
    return output


def normalize_match_details(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    players = []
    drafts = []
    purchases = []
    for match in rows:
        match_id = match.get("match_id")
        if not match_id:
            continue
        for player in match.get("players") or []:
            hero_id = player.get("hero_id")
            if not hero_id:
                continue
            players.append(
                {
                    "match_id": int(match_id),
                    "account_id": player.get("account_id"),
                    "player_slot": player.get("player_slot"),
                    "is_radiant": bool(player.get("isRadiant"))
                    if player.get("isRadiant") is not None
                    else int(player.get("player_slot") or 0) < 128,
                    "hero_id": int(hero_id),
                    "lane": player.get("lane"),
                    "lane_role": player.get("lane_role"),
                    "rank_tier": player.get("rank_tier"),
                    "kills": player.get("kills"),
                    "deaths": player.get("deaths"),
                    "assists": player.get("assists"),
                    "gold_per_min": player.get("gold_per_min"),
                    "xp_per_min": player.get("xp_per_min"),
                    "win": player.get("win"),
                    "patch": player.get("patch") or match.get("patch"),
                }
            )
            for purchase in player.get("purchase_log") or []:
                item_key = purchase.get("key")
                if not item_key:
                    continue
                purchases.append(
                    {
                        "match_id": int(match_id),
                        "account_id": player.get("account_id"),
                        "is_radiant": bool(player.get("isRadiant"))
                        if player.get("isRadiant") is not None
                        else int(player.get("player_slot") or 0) < 128,
                        "hero_id": int(hero_id),
                        "item_key": str(item_key),
                        "time": purchase.get("time"),
                        "patch": player.get("patch") or match.get("patch"),
                    }
                )
        for pick in match.get("picks_bans") or []:
            drafts.append(
                {
                    "match_id": int(match_id),
                    "is_pick": bool(pick.get("is_pick")),
                    "hero_id": pick.get("hero_id"),
                    "team": pick.get("team"),
                    "order": pick.get("order"),
                }
            )
    return players, drafts, purchases


def build_pair_stats(player_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_match: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in player_rows:
        by_match[int(row["match_id"])].append(row)

    pair_counts: Counter[tuple[int, int, str]] = Counter()
    pair_wins: Counter[tuple[int, int, str]] = Counter()
    for rows in by_match.values():
        for a in rows:
            for b in rows:
                if a["hero_id"] == b["hero_id"]:
                    continue
                relation = "ally" if a["is_radiant"] == b["is_radiant"] else "enemy"
                key = (int(a["hero_id"]), int(b["hero_id"]), relation)
                pair_counts[key] += 1
                if a.get("win") == 1:
                    pair_wins[key] += 1

    return [
        {
            "hero_id": hero_id,
            "other_hero_id": other_id,
            "relation": relation,
            "games": games,
            "wins": pair_wins[(hero_id, other_id, relation)],
            "win_rate": pair_wins[(hero_id, other_id, relation)] / games,
        }
        for (hero_id, other_id, relation), games in pair_counts.items()
        if games
    ]


def _time_bucket(seconds: int | None) -> str:
    if seconds is None:
        return "unknown"
    if seconds < 600:
        return "0-10m"
    if seconds < 1200:
        return "10-20m"
    if seconds < 1800:
        return "20-30m"
    if seconds < 2400:
        return "30-40m"
    return "40m+"


def build_hero_build_stats(purchase_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, str], list[int]] = defaultdict(list)
    for row in purchase_rows:
        hero_id = row.get("hero_id")
        item_key = row.get("item_key")
        if not hero_id or not item_key:
            continue
        time = row.get("time")
        try:
            time = int(time)
        except (TypeError, ValueError):
            time = None
        grouped[(int(hero_id), str(item_key), _time_bucket(time))].append(time or 0)

    return [
        {
            "hero_id": hero_id,
            "item_key": item_key,
            "time_bucket": time_bucket,
            "purchases": len(times),
            "median_time": float(median(times)),
        }
        for (hero_id, item_key, time_bucket), times in grouped.items()
    ]


def normalize_all(raw_dir: Path, parquet_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}

    hero_stats = normalize_hero_stats(
        read_jsonl(raw_dir / "reference" / "opendota_hero_stats.jsonl")
    )
    counts["dim_hero"] = write_parquet(
        parquet_dir / "dim_hero.parquet", hero_stats, schema=SCHEMAS["dim_hero"]
    )

    counts["dim_item"] = write_parquet(
        parquet_dir / "dim_item.parquet",
        normalize_items(read_jsonl(raw_dir / "reference" / "opendota_constants_items.jsonl")),
        schema=SCHEMAS["dim_item"],
    )
    counts["dim_patch"] = write_parquet(
        parquet_dir / "dim_patch.parquet",
        normalize_patches(read_jsonl(raw_dir / "reference" / "opendota_constants_patch.jsonl")),
        schema=SCHEMAS["dim_patch"],
    )
    counts["dim_league"] = write_parquet(
        parquet_dir / "dim_league.parquet",
        normalize_leagues(read_jsonl(raw_dir / "reference" / "opendota_leagues.jsonl")),
        schema=SCHEMAS["dim_league"],
    )

    pro_matches = normalize_match_summaries(
        read_jsonl(raw_dir / "matches" / "opendota_pro_matches.jsonl"), source="opendota_pro"
    )
    league_matches = normalize_match_summaries(
        read_jsonl(raw_dir / "matches" / "opendota_league_matches.jsonl"),
        source="opendota_league",
    )
    public_matches = normalize_match_summaries(
        read_jsonl(raw_dir / "matches" / "opendota_public_matches.jsonl"), source="opendota_public"
    )
    targeted_matches = normalize_match_summaries(
        read_jsonl(raw_dir / "matches" / "opendota_targeted_matches.jsonl"),
        source="opendota_targeted",
    )
    counts["fact_match"] = write_parquet(
        parquet_dir / "fact_match.parquet",
        dedupe_match_summaries(pro_matches + league_matches + public_matches + targeted_matches),
        schema=SCHEMAS["fact_match"],
    )

    match_details = read_jsonl(raw_dir / "matches" / "opendota_match_details.jsonl")
    players, drafts, purchases = normalize_match_details(match_details)
    counts["fact_player_match"] = write_parquet(
        parquet_dir / "fact_player_match.parquet", players, schema=SCHEMAS["fact_player_match"]
    )
    counts["fact_draft_pickban"] = write_parquet(
        parquet_dir / "fact_draft_pickban.parquet", drafts, schema=SCHEMAS["fact_draft_pickban"]
    )
    counts["fact_item_purchase"] = write_parquet(
        parquet_dir / "fact_item_purchase.parquet",
        purchases,
        schema=SCHEMAS["fact_item_purchase"],
    )
    counts["fact_hero_pair_stats"] = write_parquet(
        parquet_dir / "fact_hero_pair_stats.parquet",
        build_pair_stats(players),
        schema=SCHEMAS["fact_hero_pair_stats"],
    )
    counts["fact_hero_build_stats"] = write_parquet(
        parquet_dir / "fact_hero_build_stats.parquet",
        build_hero_build_stats(purchases),
        schema=SCHEMAS["fact_hero_build_stats"],
    )

    patch_changes = read_jsonl(raw_dir / "patches" / "valve_patch_changes.jsonl")
    counts["doc_patch_change"] = write_parquet(
        parquet_dir / "doc_patch_change.parquet",
        patch_changes,
        schema=SCHEMAS["doc_patch_change"],
    )
    return counts
