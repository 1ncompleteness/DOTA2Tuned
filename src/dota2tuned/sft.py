from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from dota2tuned.rag import Retriever
from dota2tuned.recommend import DraftRecommender
from dota2tuned.schemas import DraftInput
from dota2tuned.storage import read_parquet

ROLES = ["carry", "mid", "offlane", "soft support", "hard support"]


def _pct(value: Any) -> str:
    if value is None:
        return "unknown"
    return f"{float(value) * 100:.1f}%"


def _minutes(seconds: Any) -> str:
    if seconds is None:
        return "unknown"
    return f"{max(float(seconds), 0.0) / 60:.1f} min"


def _clip(value: Any, limit: int = 320) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _append(examples: list[dict[str, Any]], seen: set[str], prompt: str, answer: Any) -> None:
    if prompt in seen:
        return
    seen.add(prompt)
    examples.append(
        {
            "messages": [
                {"role": "user", "content": prompt},
                {
                    "role": "assistant",
                    "content": json.dumps(answer, sort_keys=True),
                },
            ]
        }
    )


def _hero_names(heroes: pl.DataFrame) -> dict[int, str]:
    if heroes.is_empty():
        return {}
    return {
        int(row["hero_id"]): str(row.get("hero_name") or f"Hero {row['hero_id']}")
        for row in heroes.iter_rows(named=True)
    }


def create_sft_examples(
    parquet_dir: Path, rag_dir: Path, output_path: Path, *, limit: int = 100
) -> int:
    recommender = DraftRecommender(parquet_dir)
    retriever = Retriever(rag_dir)
    heroes = read_parquet(parquet_dir / "dim_hero.parquet")
    build_stats = read_parquet(parquet_dir / "fact_hero_build_stats.parquet")
    pair_stats = read_parquet(parquet_dir / "fact_hero_pair_stats.parquet")
    patch_changes = read_parquet(parquet_dir / "doc_patch_change.parquet")
    names = _hero_names(heroes)
    examples: list[dict[str, Any]] = []
    seen: set[str] = set()
    if not recommender.ready():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("")
        return 0

    draft_cap = max(1, int(limit * 0.30))
    hero_cap = max(1, int(limit * 0.20))
    build_cap = max(1, int(limit * 0.20))
    pair_cap = max(1, int(limit * 0.15))
    patch_cap = max(1, limit - draft_cap - hero_cap - build_cap - pair_cap)

    top_hero_ids = (
        heroes.sort(["pro_pick", "pro_win_rate"], descending=[True, True])
        .head(32)
        .get_column("hero_id")
        .to_list()
        if not heroes.is_empty()
        else []
    )
    seed_drafts = [DraftInput(role=role, scope="pro") for role in ROLES]
    for idx, hero_id in enumerate(top_hero_ids):
        if len(top_hero_ids) <= 1:
            break
        ally = top_hero_ids[(idx + 2) % len(top_hero_ids)]
        enemy = top_hero_ids[(idx + 1) % len(top_hero_ids)]
        seed_drafts.append(
            DraftInput(
                allied_heroes=[int(hero_id)],
                enemy_heroes=[int(enemy)],
                role=ROLES[idx % len(ROLES)],
                scope="pro",
            )
        )
        seed_drafts.append(
            DraftInput(
                allied_heroes=[int(hero_id), int(ally)],
                enemy_heroes=[int(enemy)],
                role=ROLES[(idx + 1) % len(ROLES)],
                scope="pro",
            )
        )

    added = 0
    for draft in seed_drafts:
        recs = recommender.recommend(draft, limit=5)
        if not recs:
            continue
        evidence = retriever.search(" ".join(rec.hero_name for rec in recs), limit=5)
        answer = {
            "task": "draft_recommendation",
            "recommendations": [rec.model_dump() for rec in recs],
            "evidence": evidence,
        }
        context = ""
        if draft.allied_heroes:
            context += f" Allied heroes: {draft.allied_heroes}."
        if draft.enemy_heroes:
            context += f" Enemy heroes: {draft.enemy_heroes}."
        before = len(examples)
        _append(
            examples,
            seen,
            f"Suggest {draft.role} heroes for a current pro patch Dota draft.{context}",
            answer,
        )
        if len(examples) > before:
            added += 1
        if len(examples) >= limit or added >= draft_cap:
            break

    if len(examples) < limit and not heroes.is_empty():
        added = 0
        for row in heroes.sort("pro_pick", descending=True).head(limit).iter_rows(named=True):
            hero_name = row.get("hero_name") or f"Hero {row['hero_id']}"
            answer = {
                "task": "hero_meta",
                "hero": hero_name,
                "hero_id": row.get("hero_id"),
                "pro_pick": row.get("pro_pick"),
                "pro_win": row.get("pro_win"),
                "pro_win_rate": _pct(row.get("pro_win_rate")),
                "roles": row.get("roles"),
                "sources": ["OpenDota heroStats"],
                "caveat": "Use current-patch match details for draft-specific decisions.",
            }
            before = len(examples)
            _append(
                examples,
                seen,
                f"What is the current pro meta read on {hero_name}?",
                answer,
            )
            if len(examples) > before:
                added += 1
            if len(examples) >= limit or added >= hero_cap:
                break

    if len(examples) < limit and not build_stats.is_empty():
        added = 0
        grouped = (
            build_stats.group_by("hero_id")
            .agg(pl.col("purchases").sum().alias("total_purchases"))
            .sort("total_purchases", descending=True)
            .head(limit * 2)
        )
        for row in grouped.iter_rows(named=True):
            hero_id = int(row["hero_id"])
            hero_name = names.get(hero_id, f"Hero {hero_id}")
            item_rows = (
                build_stats.filter(pl.col("hero_id") == hero_id)
                .sort("purchases", descending=True)
                .head(8)
                .iter_rows(named=True)
            )
            items = [
                {
                    "item_key": item.get("item_key"),
                    "time_bucket": item.get("time_bucket"),
                    "purchases": item.get("purchases"),
                    "median_time": _minutes(item.get("median_time")),
                }
                for item in item_rows
            ]
            before = len(examples)
            _append(
                examples,
                seen,
                f"What observed item timings should I consider for {hero_name}?",
                {
                    "task": "build_timing",
                    "hero": hero_name,
                    "hero_id": hero_id,
                    "observed_items": items,
                    "sources": ["OpenDota match details", "normalized item purchase stats"],
                    "caveat": "These are observed timings, not a mandatory build order.",
                },
            )
            if len(examples) > before:
                added += 1
            if len(examples) >= limit or added >= build_cap:
                break

    if len(examples) < limit and not pair_stats.is_empty():
        added = 0
        pair_rows = (
            pair_stats.filter(pl.col("games") > 1)
            .sort(["games", "win_rate"], descending=[True, True])
            .head(limit * 4)
            .iter_rows(named=True)
        )
        for row in pair_rows:
            hero_id = int(row["hero_id"])
            other_id = int(row["other_hero_id"])
            relation = row.get("relation")
            hero_name = names.get(hero_id, f"Hero {hero_id}")
            other_name = names.get(other_id, f"Hero {other_id}")
            prompt_relation = "with" if relation == "ally" else "against"
            before = len(examples)
            _append(
                examples,
                seen,
                f"How does {hero_name} perform {prompt_relation} {other_name}?",
                {
                    "task": "pair_stats",
                    "hero": hero_name,
                    "other_hero": other_name,
                    "relation": relation,
                    "games": row.get("games"),
                    "wins": row.get("wins"),
                    "win_rate": _pct(row.get("win_rate")),
                    "sources": ["normalized pair stats"],
                    "caveat": "Pair stats are directional and can be noisy at low sample size.",
                },
            )
            if len(examples) > before:
                added += 1
            if len(examples) >= limit or added >= pair_cap:
                break

    if len(examples) < limit and not patch_changes.is_empty():
        added = 0
        for row in patch_changes.head(limit * 4).iter_rows(named=True):
            text = _clip(row.get("text"))
            if not text:
                continue
            patch = row.get("patch")
            section = row.get("section")
            before = len(examples)
            _append(
                examples,
                seen,
                (
                    f"What does this patch {patch} {section} note mean for Dota analysis: "
                    f"{_clip(text, 140)}"
                ),
                {
                    "task": "patch_note",
                    "patch": patch,
                    "section": section,
                    "subject_type": row.get("subject_type"),
                    "subject_id": row.get("subject_id"),
                    "change": text,
                    "sources": ["Valve patch notes"],
                },
            )
            if len(examples) > before:
                added += 1
            if len(examples) >= limit or added >= patch_cap:
                break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        for example in examples:
            fh.write(json.dumps(example) + "\n")
    return len(examples)
