from __future__ import annotations

import re
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from dota2tuned.storage import read_parquet


def _clean(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def build_documents(parquet_dir: Path) -> list[dict[str, str]]:
    docs: list[dict[str, str]] = []

    patch_changes = read_parquet(parquet_dir / "doc_patch_change.parquet")
    for row in patch_changes.iter_rows(named=True):
        text = _clean(row.get("text"))
        if text:
            docs.append(
                {
                    "id": f"patch:{row.get('patch')}:{len(docs)}",
                    "kind": "patch_change",
                    "patch": _clean(row.get("patch")),
                    "text": f"Patch {row.get('patch')} {row.get('section')}: {text}",
                    "source": "Valve patch notes",
                }
            )

    heroes = read_parquet(parquet_dir / "dim_hero.parquet")
    hero_names = {
        int(row["hero_id"]): str(row.get("hero_name") or f"Hero {row['hero_id']}")
        for row in heroes.iter_rows(named=True)
        if row.get("hero_id") is not None
    }
    for row in heroes.iter_rows(named=True):
        pro_pick = int(row.get("pro_pick") or 0)
        pro_win = int(row.get("pro_win") or 0)
        win_rate = row.get("pro_win_rate")
        win_rate_text = f"{round(float(win_rate) * 100, 1)} percent" if win_rate else "unknown"
        docs.append(
            {
                "id": f"hero:{row.get('hero_id')}",
                "kind": "stat_card",
                "patch": "current",
                "text": (
                    f"{row.get('hero_name')} current pro stat card: "
                    f"{pro_pick} pro picks, {pro_win} wins, "
                    f"{win_rate_text} win rate. "
                    f"Roles: {row.get('roles')}."
                ),
                "source": "OpenDota heroStats",
            }
        )

    items = read_parquet(parquet_dir / "dim_item.parquet")
    item_names = {}
    for row in items.iter_rows(named=True):
        item_key = _clean(row.get("item_key"))
        if not item_key:
            continue
        item_name = _clean(row.get("item_name")) or item_key.replace("_", " ").title()
        item_names[item_key] = item_name
        parts = [
            f"{item_name} item card.",
            f"Key: {item_key}.",
        ]
        if row.get("cost") is not None:
            parts.append(f"Cost: {row.get('cost')}.")
        if _clean(row.get("notes")):
            parts.append(f"Notes: {_clean(row.get('notes'))}.")
        if _clean(row.get("attrib")):
            parts.append(f"Attributes: {_clean(row.get('attrib'))}.")
        docs.append(
            {
                "id": f"item:{item_key}",
                "kind": "item_card",
                "patch": "current",
                "text": " ".join(parts),
                "source": "OpenDota item constants",
            }
        )

    build_stats = read_parquet(parquet_dir / "fact_hero_build_stats.parquet")
    if not build_stats.is_empty():
        for row in (
            build_stats.sort("purchases", descending=True)
            .head(3000)
            .iter_rows(named=True)
        ):
            hero_id = int(row.get("hero_id") or 0)
            item_key = _clean(row.get("item_key"))
            if not hero_id or not item_key:
                continue
            docs.append(
                {
                    "id": (
                        f"build:{hero_id}:{row.get('role')}:{item_key}:"
                        f"{row.get('time_bucket')}"
                    ),
                    "kind": "item_timing",
                    "patch": "current",
                    "text": (
                        f"{hero_names.get(hero_id, f'Hero {hero_id}')} observed "
                        f"{item_names.get(item_key, item_key.replace('_', ' ').title())} "
                        f"timing for role {row.get('role')}: bucket {row.get('time_bucket')}, "
                        f"{row.get('purchases')} purchases, median time "
                        f"{row.get('median_time')} seconds."
                    ),
                    "source": "OpenDota normalized item purchase stats",
                }
            )

    abilities = read_parquet(parquet_dir / "dim_ability.parquet")
    ability_names = {
        int(row["ability_id"]): _clean(row.get("ability_name"))
        or _clean(row.get("ability_key"))
        or f"Ability {row['ability_id']}"
        for row in abilities.iter_rows(named=True)
        if row.get("ability_id") is not None
    }
    skill_stats = read_parquet(parquet_dir / "fact_hero_skill_builds.parquet")
    if not skill_stats.is_empty():
        for row in (
            skill_stats.sort("picks", descending=True).head(3000).iter_rows(named=True)
        ):
            hero_id = int(row.get("hero_id") or 0)
            ability_id = int(row.get("ability_id") or 0)
            if not hero_id or not ability_id:
                continue
            docs.append(
                {
                    "id": (
                        f"skill:{hero_id}:{row.get('role')}:{ability_id}:"
                        f"{row.get('pick_order')}"
                    ),
                    "kind": "skill_build",
                    "patch": "current",
                    "text": (
                        f"{hero_names.get(hero_id, f'Hero {hero_id}')} commonly levels "
                        f"{ability_names.get(ability_id, f'Ability {ability_id}')} at "
                        f"skill order {row.get('pick_order')} for role {row.get('role')}, "
                        f"observed {row.get('picks')} times."
                    ),
                    "source": "OpenDota normalized ability upgrade stats",
                }
            )

    stratz_docs = read_parquet(parquet_dir / "doc_stratz_match.parquet")
    for row in stratz_docs.iter_rows(named=True):
        text = _clean(row.get("text"))
        match_id = row.get("match_id")
        if not text or match_id is None:
            continue
        docs.append(
            {
                "id": f"stratz_match:{match_id}",
                "kind": "stratz_match",
                "patch": _clean(row.get("patch")) or "current",
                "text": text,
                "source": "STRATZ match details",
            }
        )
    return docs


def build_index(parquet_dir: Path, rag_dir: Path) -> int:
    rag_dir.mkdir(parents=True, exist_ok=True)
    docs = build_documents(parquet_dir)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform([doc["text"] for doc in docs]) if docs else None
    joblib.dump(
        {"docs": docs, "vectorizer": vectorizer, "matrix": matrix}, rag_dir / "tfidf.joblib"
    )
    return len(docs)


class Retriever:
    def __init__(self, rag_dir: Path) -> None:
        self.path = rag_dir / "tfidf.joblib"
        self.payload = joblib.load(self.path) if self.path.exists() else None

    def search(
        self, query: str, *, patch: str | None = None, limit: int = 5
    ) -> list[dict[str, str]]:
        if not self.payload or self.payload["matrix"] is None:
            return []
        docs: list[dict[str, str]] = self.payload["docs"]
        vectorizer: TfidfVectorizer = self.payload["vectorizer"]
        matrix = self.payload["matrix"]
        query_vec = vectorizer.transform([query])
        scores = cosine_similarity(query_vec, matrix).ravel()
        ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
        output = []
        for idx, score in ranked:
            doc = docs[idx]
            if patch and patch != "current" and doc.get("patch") not in {patch, "current"}:
                continue
            output.append({**doc, "score": f"{score:.4f}"})
            if len(output) >= limit:
                break
        return output
