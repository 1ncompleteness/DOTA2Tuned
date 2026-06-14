from __future__ import annotations

from typing import Any

from dota2tuned.http import HttpJsonClient, RateLimit

MATCH_QUERY = """
query Match($id: Long!) {
  match(id: $id) {
    id
    didRadiantWin
    durationSeconds
    startDateTime
    gameMode
    lobbyType
    players {
      steamAccountId
      heroId
      position
      lane
      isRadiant
      kills
      deaths
      assists
      level
      goldPerMinute
      experiencePerMinute
      imp
    }
    pickBans {
      heroId
      isPick
      order
      bannedHeroId
      team
    }
  }
}
"""


class StratzClient:
    def __init__(self, token: str | None) -> None:
        headers = {"User-Agent": "DOTA2Tuned Hackathon"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.http = HttpJsonClient(
            "https://api.stratz.com",
            headers=headers,
            rate_limit=RateLimit(calls=250, period_seconds=60),
            timeout=45,
        )

    def close(self) -> None:
        self.http.close()

    def execute(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = self.http.post_json(
            "/graphql", json_body={"query": query, "variables": variables or {}}
        )
        if payload.get("errors"):
            raise RuntimeError(payload["errors"])
        return payload["data"]

    def match(self, match_id: int) -> dict[str, Any]:
        return self.execute(MATCH_QUERY, {"id": match_id})["match"]
