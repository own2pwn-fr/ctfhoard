"""Shared HTTP client.

A single configured ``httpx.Client`` factory so every connector sends a descriptive
User-Agent (several sources 403 generic fetchers), retries transient failures with
backoff, and can be pointed through an optional rate limiter. Connectors should not
build raw clients — they get one from here.
"""

from __future__ import annotations

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ctfhoard.ratelimit import RateLimiter

USER_AGENT = (
    "ctfhoard/0.1 (+https://github.com/own2pwn-fr/ctfhoard) "
    "open CTF corpus aggregator; contact contact@own2pwn.fr"
)


def make_client(
    *,
    timeout: float = 30.0,
    headers: dict[str, str] | None = None,
    follow_redirects: bool = True,
) -> httpx.Client:
    """Build a configured httpx client with our UA and sane timeouts."""
    base_headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}
    if headers:
        base_headers.update(headers)
    return httpx.Client(
        headers=base_headers,
        timeout=timeout,
        follow_redirects=follow_redirects,
    )


class PoliteClient:
    """A client wrapper that rate-limits and retries every request.

    Wrap a source's client with its own :class:`RateLimiter` (e.g. 10 s crawl-delay
    for CTFtime) and get transparent backoff on 429/5xx/network errors.
    """

    def __init__(self, client: httpx.Client, limiter: RateLimiter | None = None) -> None:
        self._client = client
        self._limiter = limiter

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def get(self, url: str, **kwargs) -> httpx.Response:
        if self._limiter is not None:
            self._limiter.acquire()
        resp = self._client.get(url, **kwargs)
        # Retry on rate-limit / server errors; 4xx (except 429) fail fast.
        if resp.status_code == 429 or resp.status_code >= 500:
            resp.raise_for_status()
        return resp

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> PoliteClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
