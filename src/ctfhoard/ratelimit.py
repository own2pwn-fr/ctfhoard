"""Polite rate limiting for source crawlers.

Every source has etiquette we must honor: CTFtime declares ``Crawl-delay: 10`` and
an ``ai-train=no`` reference-only signal; GitHub caps search at 30 req/min and core
at 5000 req/hr. A shared token-bucket + minimum-interval limiter keeps every
connector well-behaved without each reimplementing backoff.
"""

from __future__ import annotations

import threading
import time


class RateLimiter:
    """Token bucket with an optional hard minimum interval between acquisitions.

    * ``rate`` / ``per`` define the refill (e.g. 30 tokens per 60 s = GitHub search).
    * ``min_interval`` enforces a floor between calls (e.g. 10 s CTFtime crawl-delay)
      regardless of available tokens.

    Thread-safe and monotonic-clock based so it is unaffected by wall-clock jumps.
    """

    def __init__(
        self,
        rate: float = 1.0,
        per: float = 1.0,
        *,
        burst: float | None = None,
        min_interval: float = 0.0,
    ) -> None:
        self.rate = rate
        self.per = per
        self.capacity = burst if burst is not None else rate
        self.min_interval = min_interval
        self._tokens = self.capacity
        self._last_refill = time.monotonic()
        self._last_acquire = 0.0
        self._lock = threading.Lock()

    def _refill(self, now: float) -> None:
        elapsed = now - self._last_refill
        self._tokens = min(self.capacity, self._tokens + elapsed * (self.rate / self.per))
        self._last_refill = now

    def acquire(self, tokens: float = 1.0) -> float:
        """Block until ``tokens`` are available and the min interval has elapsed.

        Returns the total time slept, for observability/testing.
        """
        slept = 0.0
        while True:
            with self._lock:
                now = time.monotonic()
                self._refill(now)
                wait_interval = max(0.0, self.min_interval - (now - self._last_acquire))
                if self._tokens >= tokens and wait_interval <= 0:
                    self._tokens -= tokens
                    self._last_acquire = now
                    return slept
                if self._tokens < tokens:
                    deficit = tokens - self._tokens
                    wait_tokens = deficit * (self.per / self.rate)
                else:
                    wait_tokens = 0.0
                wait = max(wait_interval, wait_tokens)
            time.sleep(wait)
            slept += wait
