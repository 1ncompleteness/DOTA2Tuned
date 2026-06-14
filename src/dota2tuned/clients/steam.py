from __future__ import annotations

from typing import Any

from dota2tuned.http import HttpJsonClient, RateLimit


class SteamClient:
    def __init__(self, api_key: str | None) -> None:
        params = {"key": api_key} if api_key else {}
        self.http = HttpJsonClient(
            "https://api.steampowered.com",
            params=params,
            rate_limit=RateLimit(calls=60, period_seconds=60),
        )

    def close(self) -> None:
        self.http.close()

    def get_match_details(self, match_id: int) -> dict[str, Any]:
        return self.http.get(
            "/IDOTA2Match_570/GetMatchDetails/v1/",
            params={"match_id": match_id, "format": "json"},
        )["result"]

    def get_match_history_by_sequence_num(
        self, start_at_match_seq_num: int, matches_requested: int = 100
    ) -> dict[str, Any]:
        return self.http.get(
            "/IDOTA2Match_570/GetMatchHistoryBySequenceNum/v1/",
            params={
                "start_at_match_seq_num": start_at_match_seq_num,
                "matches_requested": matches_requested,
                "format": "json",
            },
        )["result"]

    def get_heroes(self, language: str = "en_us") -> dict[str, Any]:
        return self.http.get(
            "/IEconDOTA2_570/GetHeroes/v1/",
            params={"language": language, "format": "json"},
        )["result"]
