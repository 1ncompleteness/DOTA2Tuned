from __future__ import annotations

from pathlib import Path
from typing import Any

from dota2tuned.clients import OpenDotaClient, StratzClient, ValvePatchClient
from dota2tuned.clients.valve import flatten_patch_notes
from dota2tuned.config import Settings
from dota2tuned.storage import write_jsonl


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
            for row in patch_rows:
                version = row.get("name")
                if not version:
                    continue
                try:
                    payload = valve.patch_notes(version)
                except Exception:
                    continue
                changes.extend(flatten_patch_notes(payload))
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
            details = []
            stratz_details = []
            for row in selected:
                match_id = int(row["match_id"])
                try:
                    details.append(od.match(match_id))
                except Exception as exc:
                    details.append({"match_id": match_id, "error": str(exc), "source": "opendota"})
                if stratz and len(stratz_details) < stratz_limit:
                    try:
                        stratz_details.append(stratz.match(match_id))
                    except Exception as exc:
                        stratz_details.append(
                            {"id": match_id, "error": str(exc), "source": "stratz"}
                        )

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
                    self.raw_dir / "matches" / "opendota_match_details.jsonl", details
                ),
            }
            if stratz_details:
                counts["stratz_match_details"] = write_jsonl(
                    self.raw_dir / "matches" / "stratz_match_details.jsonl", stratz_details
                )
            return counts
        finally:
            od.close()
            if stratz:
                stratz.close()


def raw_path(settings: Settings, *parts: str) -> Path:
    return settings.raw_data_dir.joinpath(*parts)
