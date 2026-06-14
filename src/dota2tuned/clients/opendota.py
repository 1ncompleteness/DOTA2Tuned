from __future__ import annotations

from typing import Any

from dota2tuned.http import HttpJsonClient, RateLimit


class OpenDotaClient:
    def __init__(self, api_key: str | None = None) -> None:
        params = {"api_key": api_key} if api_key else {}
        self.http = HttpJsonClient(
            "https://api.opendota.com/api",
            params=params,
            rate_limit=RateLimit(calls=60, period_seconds=60),
        )

    def close(self) -> None:
        self.http.close()

    def metadata(self) -> dict[str, Any]:
        return self.http.get("/metadata")

    def constants(self, resource: str) -> Any:
        return self.http.get(f"/constants/{resource}")

    def hero_stats(self) -> list[dict[str, Any]]:
        return self.http.get("/heroStats")

    def pro_matches(self) -> list[dict[str, Any]]:
        return self.http.get("/proMatches")

    def public_matches(self) -> list[dict[str, Any]]:
        return self.http.get("/publicMatches")

    def parsed_matches(self) -> list[dict[str, Any]]:
        return self.http.get("/parsedMatches")

    def leagues(self) -> list[dict[str, Any]]:
        return self.http.get("/leagues")

    def league_matches(self, league_id: int) -> list[dict[str, Any]]:
        return self.http.get(f"/leagues/{league_id}/matches")

    def match(self, match_id: int) -> dict[str, Any]:
        return self.http.get(f"/matches/{match_id}")

    def hero_matchups(self, hero_id: int) -> list[dict[str, Any]]:
        return self.http.get(f"/heroes/{hero_id}/matchups")

    def hero_item_popularity(self, hero_id: int) -> dict[str, Any]:
        return self.http.get(f"/heroes/{hero_id}/itemPopularity")

    def explorer(self, sql: str) -> dict[str, Any]:
        return self.http.get("/explorer", params={"sql": sql})
