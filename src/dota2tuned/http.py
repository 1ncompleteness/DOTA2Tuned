from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


class ApiError(RuntimeError):
    pass


@dataclass
class RateLimit:
    calls: int
    period_seconds: float


class TokenBucket:
    def __init__(self, limit: RateLimit) -> None:
        self.limit = limit
        self._timestamps: list[float] = []

    def wait(self) -> None:
        now = time.monotonic()
        self._timestamps = [t for t in self._timestamps if now - t < self.limit.period_seconds]
        if len(self._timestamps) >= self.limit.calls:
            sleep_for = self.limit.period_seconds - (now - self._timestamps[0])
            if sleep_for > 0:
                time.sleep(sleep_for)
        self._timestamps.append(time.monotonic())


class JsonCache:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> Any | None:
        path = self.root / f"{key}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def set(self, key: str, value: Any) -> None:
        path = self.root / f"{key}.json"
        path.write_text(json.dumps(value, sort_keys=True))


class HttpJsonClient:
    def __init__(
        self,
        base_url: str,
        *,
        headers: dict[str, str] | None = None,
        params: dict[str, str] | None = None,
        timeout: float = 30.0,
        rate_limit: RateLimit | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.params = params or {}
        self.bucket = TokenBucket(rate_limit) if rate_limit else None
        self.client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout),
            follow_redirects=True,
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> HttpJsonClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, ApiError)),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def get(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        if self.bucket:
            self.bucket.wait()
        merged = {**self.params, **(params or {})}
        response = self.client.get(path, params=merged)
        if response.status_code in {429, 500, 502, 503, 504}:
            raise ApiError(f"GET {path} returned retryable {response.status_code}")
        if response.status_code >= 400:
            raise ApiError(f"GET {path} returned {response.status_code}: {response.text[:300]}")
        return response.json()

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, ApiError)),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def post_json(self, path: str, *, json_body: dict[str, Any]) -> Any:
        if self.bucket:
            self.bucket.wait()
        response = self.client.post(path, json=json_body)
        if response.status_code in {429, 500, 502, 503, 504}:
            raise ApiError(f"POST {path} returned retryable {response.status_code}")
        if response.status_code >= 400:
            raise ApiError(f"POST {path} returned {response.status_code}: {response.text[:300]}")
        return response.json()
