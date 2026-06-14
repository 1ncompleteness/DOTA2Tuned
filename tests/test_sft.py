import json
from pathlib import Path

from dota2tuned.rag import build_index
from dota2tuned.sft import create_sft_examples
from dota2tuned.storage import write_parquet


def test_create_sft_examples_draws_from_multiple_sources(tmp_path: Path):
    parquet = tmp_path / "parquet"
    rag = tmp_path / "rag"
    output = tmp_path / "sft.jsonl"

    write_parquet(
        parquet / "dim_hero.parquet",
        [
            {
                "hero_id": hero_id,
                "hero_name": f"Hero {hero_id}",
                "roles": ["Carry"],
                "pro_pick": 100 + hero_id,
                "pro_win": 50 + hero_id,
                "pro_win_rate": 0.5 + hero_id / 1000,
            }
            for hero_id in range(1, 8)
        ],
    )
    write_parquet(
        parquet / "fact_hero_pair_stats.parquet",
        [
            {
                "hero_id": 1,
                "other_hero_id": 2,
                "relation": "ally",
                "games": 3,
                "wins": 2,
                "win_rate": 2 / 3,
            },
            {
                "hero_id": 3,
                "other_hero_id": 4,
                "relation": "enemy",
                "games": 4,
                "wins": 1,
                "win_rate": 0.25,
            },
        ],
    )
    write_parquet(
        parquet / "fact_hero_build_stats.parquet",
        [
            {
                "hero_id": 1,
                "item_key": "blink",
                "time_bucket": "10-20m",
                "purchases": 9,
                "median_time": 780.0,
            },
            {
                "hero_id": 2,
                "item_key": "bkb",
                "time_bucket": "20-30m",
                "purchases": 7,
                "median_time": 1500.0,
            },
        ],
    )
    write_parquet(
        parquet / "doc_patch_change.parquet",
        [
            {
                "patch": "7.41",
                "patch_timestamp": 1,
                "section": "Heroes",
                "subject_type": "hero",
                "subject_id": 1,
                "text": "Hero 1 base armor increased by 1.",
                "indent_level": 1,
                "aghanims": None,
                "icon": None,
            }
        ],
    )

    build_index(parquet, rag)
    count = create_sft_examples(parquet, rag, output, limit=20)

    assert count > 3
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert count == len(rows)
    for row in rows:
        messages = row["messages"]
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert json.loads(messages[1]["content"])
