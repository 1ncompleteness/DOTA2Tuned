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
