from pathlib import Path

from dota2tuned.normalize import normalize_all
from dota2tuned.storage import read_parquet, write_jsonl


def test_normalize_all_builds_core_tables(tmp_path: Path):
    raw = tmp_path / "raw"
    parquet = tmp_path / "parquet"

    write_jsonl(
        raw / "reference" / "opendota_hero_stats.jsonl",
        [
            {
                "id": 1,
                "localized_name": "Anti-Mage",
                "primary_attr": "agi",
                "attack_type": "Melee",
                "roles": ["Carry"],
                "pro_pick": 10,
                "pro_win": 6,
            },
            {
                "id": 2,
                "localized_name": "Axe",
                "primary_attr": "str",
                "attack_type": "Melee",
                "roles": ["Initiator"],
                "pro_pick": 8,
                "pro_win": 3,
            },
        ],
    )
    write_jsonl(
        raw / "reference" / "opendota_constants_items.jsonl",
        [{"id": 1, "key": "blink", "dname": "Blink Dagger", "cost": 2250}],
    )
    write_jsonl(
        raw / "reference" / "opendota_constants_patch.jsonl",
        [{"id": 1, "name": "7.41d", "date": "2026-06-04T00:00:00Z"}],
    )
    write_jsonl(
        raw / "reference" / "opendota_leagues.jsonl",
        [{"leagueid": 1, "name": "Test League", "tier": "professional"}],
    )
    write_jsonl(
        raw / "matches" / "opendota_pro_matches.jsonl",
        [{"match_id": 42, "duration": 1800, "radiant_win": True, "leagueid": 1}],
    )
    write_jsonl(raw / "matches" / "opendota_league_matches.jsonl", [])
    write_jsonl(raw / "matches" / "opendota_public_matches.jsonl", [])
    write_jsonl(
        raw / "matches" / "opendota_targeted_matches.jsonl",
        [{"match_id": 43, "duration": 1500, "radiant_win": False}],
    )
    write_jsonl(
        raw / "matches" / "opendota_match_details.jsonl",
        [
            {
                "match_id": 42,
                "patch": 1,
                "players": [
                    {
                        "player_slot": 0,
                        "hero_id": 1,
                        "win": 1,
                        "purchase_log": [{"time": 700, "key": "blink"}],
                    },
                    {"player_slot": 128, "hero_id": 2, "win": 0},
                ],
                "picks_bans": [{"is_pick": True, "hero_id": 1, "team": 0, "order": 1}],
            }
        ],
    )
    write_jsonl(raw / "patches" / "valve_patch_changes.jsonl", [])

    counts = normalize_all(raw, parquet)

    assert counts["dim_hero"] == 2
    assert counts["fact_item_purchase"] == 1
    assert counts["fact_hero_build_stats"] == 1
    assert read_parquet(parquet / "fact_match.parquet").height == 2
    build_row = read_parquet(parquet / "fact_hero_build_stats.parquet").row(0, named=True)
    assert build_row["item_key"] == "blink"
    assert build_row["time_bucket"] == "10-20m"
