from __future__ import annotations

from typing import Any

from dota2tuned.http import HttpJsonClient, RateLimit


class ValvePatchClient:
    def __init__(self) -> None:
        self.http = HttpJsonClient(
            "https://www.dota2.com",
            rate_limit=RateLimit(calls=30, period_seconds=60),
        )

    def close(self) -> None:
        self.http.close()

    def patch_notes(self, version: str, language: str = "english") -> dict[str, Any]:
        return self.http.get(
            "/datafeed/patchnotes",
            params={"version": version, "language": language},
        )


def flatten_patch_notes(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    patch_number = payload.get("patch_number") or payload.get("patch_name")
    patch_timestamp = payload.get("patch_timestamp")

    def add_rows(
        section: str, subject_type: str, subject_id: int | None, notes: list[dict[str, Any]]
    ) -> None:
        for note in notes or []:
            text = note.get("note")
            if not text:
                continue
            rows.append(
                {
                    "patch": patch_number,
                    "patch_timestamp": patch_timestamp,
                    "section": section,
                    "subject_type": subject_type,
                    "subject_id": subject_id,
                    "text": text,
                    "indent_level": note.get("indent_level", 1),
                    "aghanims": note.get("aghanims"),
                    "icon": note.get("icon"),
                }
            )

    for group in payload.get("general_notes") or []:
        add_rows(group.get("title", "General"), "general", None, group.get("generic") or [])

    for item in payload.get("items") or []:
        add_rows("Items", "item", item.get("ability_id"), item.get("ability_notes") or [])

    for item in payload.get("neutral_items") or []:
        add_rows(
            "Neutral Items", "neutral_item", item.get("ability_id"), item.get("ability_notes") or []
        )

    for hero in payload.get("heroes") or []:
        hero_id = hero.get("hero_id")
        add_rows("Heroes", "hero", hero_id, hero.get("hero_notes") or [])
        add_rows("Heroes", "hero_talent", hero_id, hero.get("talent_notes") or [])
        for ability in hero.get("abilities") or []:
            add_rows(
                "Heroes", "ability", ability.get("ability_id"), ability.get("ability_notes") or []
            )

    return rows
