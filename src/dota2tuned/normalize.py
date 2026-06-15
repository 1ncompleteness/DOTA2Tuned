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
        "attrib": pl.Utf8,
        "notes": pl.Utf8,
        "lore": pl.Utf8,
    },
    "dim_ability": {
        "ability_id": pl.Int64,
        "ability_key": pl.Utf8,
        "ability_name": pl.Utf8,
        "img": pl.Utf8,
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
        "role": pl.Utf8,
        "item_key": pl.Utf8,
        "time_bucket": pl.Utf8,
        "purchases": pl.Int64,
        "median_time": pl.Float64,
    },
    "fact_hero_skill_builds": {
        "hero_id": pl.Int64,
        "role": pl.Utf8,
        "ability_id": pl.Int64,
        "pick_order": pl.Int64,
        "picks": pl.Int64,
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


def _format_attrib(attrib: Any) -> str | None:
    if not isinstance(attrib, list):
        return None
    parts = []
    for entry in attrib:
        if not isinstance(entry, dict):
            continue
        header = str(entry.get("header") or "").replace("%", "").strip()
        value = entry.get("value")
        footer = str(entry.get("footer") or "").strip()
        if not header:
            continue
        if isinstance(value, list):
            value_text = "/".join(str(item) for item in value)
        else:
            value_text = str(value) if value is not None else ""
        if "{value}" in header:
            text = header.replace("{value}", value_text)
        else:
            text = f"{header} {value_text}".strip()
        if footer:
            text = f"{text} {footer}"
        parts.append(text.strip())
    return "; ".join(part for part in parts if part) or None


def _format_notes(notes: Any) -> str | None:
    if isinstance(notes, list):
        cleaned = [str(note).strip() for note in notes if str(note).strip()]
        return " | ".join(cleaned) or None
    if isinstance(notes, str) and notes.strip():
        return notes.strip()
    return None


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
                "attrib": _format_attrib(row.get("attrib")),
                "notes": _format_notes(row.get("notes")),
                "lore": str(row.get("lore")).strip() if row.get("lore") else None,
            }
        )
    return output


def normalize_abilities(
    ability_id_rows: list[dict[str, Any]], ability_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    metadata_by_key = {
        str(row.get("key")): row for row in ability_rows if row.get("key")
    }
    output = []
    for row in ability_id_rows:
        raw_id = row.get("key")
        ability_key = row.get("value")
        try:
            ability_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        if not ability_key:
            continue
        ability_key = str(ability_key)
        metadata = metadata_by_key.get(ability_key, {})
        output.append(
            {
                "ability_id": ability_id,
                "ability_key": ability_key,
                "ability_name": metadata.get("dname") or ability_key,
                "img": metadata.get("img"),
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
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]
]:
    players = []
    drafts = []
    purchases = []
    ability_upgrades = []
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
            ability_log = player.get("ability_upgrades")
            if ability_log:
                for order, upgrade in enumerate(ability_log, start=1):
                    ability_id = upgrade.get("ability") if isinstance(upgrade, dict) else upgrade
                    if not ability_id:
                        continue
                    ability_upgrades.append(
                        {
                            "match_id": int(match_id),
                            "hero_id": int(hero_id),
                            "ability_id": int(ability_id),
                            "pick_order": order,
                            "time": upgrade.get("time") if isinstance(upgrade, dict) else None,
                        }
                    )
            else:
                for order, ability_id in enumerate(player.get("ability_upgrades_arr") or [], start=1):
                    if not ability_id:
                        continue
                    ability_upgrades.append(
                        {
                            "match_id": int(match_id),
                            "hero_id": int(hero_id),
                            "ability_id": int(ability_id),
                            "pick_order": order,
                            "time": None,
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
    return players, drafts, purchases, ability_upgrades


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


ROLE_ALL = "all"


def derive_player_roles(player_rows: list[dict[str, Any]]) -> dict[tuple[int, int], str]:
    """Map (match_id, hero_id) to a Position 1-5 role.

    Players are grouped by their lane (lane_role) within each match/side, then ranked by
    gold_per_min: the higher-farming player on safe lane is the carry (pos 1) and the other
    is hard support (pos 5); on the off lane the higher-farming player is offlane (pos 3)
    and the other is soft support (pos 4). Mid lane is always pos 2; unknown/jungle lanes
    fall back to carry.
    """
    groups: dict[tuple[int, bool, int | None], list[dict[str, Any]]] = defaultdict(list)
    for row in player_rows:
        match_id = row.get("match_id")
        hero_id = row.get("hero_id")
        if not match_id or not hero_id:
            continue
        groups[(int(match_id), bool(row.get("is_radiant")), row.get("lane_role"))].append(row)

    roles: dict[tuple[int, int], str] = {}
    for (match_id, _is_radiant, lane_role), rows in groups.items():
        ordered = sorted(rows, key=lambda r: r.get("gold_per_min") or 0, reverse=True)
        for rank, row in enumerate(ordered):
            hero_id = int(row["hero_id"])
            if lane_role == 2:
                role = "mid"
            elif lane_role == 1:
                role = "carry" if rank == 0 else "hard support"
            elif lane_role == 3:
                role = "offlane" if rank == 0 else "soft support"
            else:
                role = "carry"
            roles[(match_id, hero_id)] = role
    return roles


def build_hero_build_stats(
    purchase_rows: list[dict[str, Any]], player_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    roles = derive_player_roles(player_rows)
    grouped: dict[tuple[int, str, str, str], list[int]] = defaultdict(list)
    for row in purchase_rows:
        hero_id = row.get("hero_id")
        item_key = row.get("item_key")
        match_id = row.get("match_id")
        if not hero_id or not item_key:
            continue
        time = row.get("time")
        try:
            time = int(time)
        except (TypeError, ValueError):
            time = None
        hero_id = int(hero_id)
        bucket = _time_bucket(time)
        grouped[(hero_id, ROLE_ALL, str(item_key), bucket)].append(time or 0)
        role = roles.get((int(match_id), hero_id)) if match_id else None
        if role:
            grouped[(hero_id, role, str(item_key), bucket)].append(time or 0)

    return [
        {
            "hero_id": hero_id,
            "role": role,
            "item_key": item_key,
            "time_bucket": time_bucket,
            "purchases": len(times),
            "median_time": float(median(times)),
        }
        for (hero_id, role, item_key, time_bucket), times in grouped.items()
    ]


def build_hero_skill_stats(
    ability_rows: list[dict[str, Any]], player_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    roles = derive_player_roles(player_rows)
    counts: Counter[tuple[int, str, int, int]] = Counter()
    for row in ability_rows:
        hero_id = row.get("hero_id")
        ability_id = row.get("ability_id")
        pick_order = row.get("pick_order")
        match_id = row.get("match_id")
        if not hero_id or not ability_id or not pick_order:
            continue
        hero_id = int(hero_id)
        ability_id = int(ability_id)
        pick_order = int(pick_order)
        counts[(hero_id, ROLE_ALL, ability_id, pick_order)] += 1
        role = roles.get((int(match_id), hero_id)) if match_id else None
        if role:
            counts[(hero_id, role, ability_id, pick_order)] += 1

    return [
        {
            "hero_id": hero_id,
            "role": role,
            "ability_id": ability_id,
            "pick_order": pick_order,
            "picks": picks,
        }
        for (hero_id, role, ability_id, pick_order), picks in counts.items()
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
    counts["dim_ability"] = write_parquet(
        parquet_dir / "dim_ability.parquet",
        normalize_abilities(
            read_jsonl(raw_dir / "reference" / "opendota_constants_ability_ids.jsonl"),
            read_jsonl(raw_dir / "reference" / "opendota_constants_abilities.jsonl"),
        ),
        schema=SCHEMAS["dim_ability"],
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
    players, drafts, purchases, ability_upgrades = normalize_match_details(match_details)
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
        build_hero_build_stats(purchases, players),
        schema=SCHEMAS["fact_hero_build_stats"],
    )
    counts["fact_hero_skill_builds"] = write_parquet(
        parquet_dir / "fact_hero_skill_builds.parquet",
        build_hero_skill_stats(ability_upgrades, players),
        schema=SCHEMAS["fact_hero_skill_builds"],
    )

    patch_changes = read_jsonl(raw_dir / "patches" / "valve_patch_changes.jsonl")
    counts["doc_patch_change"] = write_parquet(
        parquet_dir / "doc_patch_change.parquet",
        patch_changes,
        schema=SCHEMAS["doc_patch_change"],
    )
    return counts
