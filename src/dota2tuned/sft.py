from __future__ import annotations

import json
from pathlib import Path

from dota2tuned.rag import Retriever
from dota2tuned.recommend import DraftRecommender
from dota2tuned.schemas import DraftInput


def create_sft_examples(
    parquet_dir: Path, rag_dir: Path, output_path: Path, *, limit: int = 100
) -> int:
    recommender = DraftRecommender(parquet_dir)
    retriever = Retriever(rag_dir)
    examples = []
    if not recommender.ready():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("")
        return 0

    seed_drafts = [
        DraftInput(allied_heroes=[], enemy_heroes=[], role="mid", scope="pro"),
        DraftInput(allied_heroes=[], enemy_heroes=[], role="carry", scope="pro"),
        DraftInput(allied_heroes=[], enemy_heroes=[], role="support", scope="pro"),
    ]
    for draft in seed_drafts:
        recs = recommender.recommend(draft, limit=5)
        if not recs:
            continue
        evidence = retriever.search(" ".join(rec.hero_name for rec in recs), limit=5)
        answer = {
            "recommendations": [rec.model_dump() for rec in recs],
            "evidence": evidence,
        }
        examples.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Suggest {draft.role} heroes for a current pro patch Dota draft."
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(answer, indent=2),
                    },
                ]
            }
        )
        if len(examples) >= limit:
            break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        for example in examples:
            fh.write(json.dumps(example) + "\n")
    return len(examples)
