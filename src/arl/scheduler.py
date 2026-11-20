"""Suite-scoped admission control and bounded transport retries."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import httpx


class EvaluationScheduler:
    def __init__(self, concurrency: int, retry_budget: int, seed: int) -> None:
        self._maximum = self._limit = concurrency
        self._budget = retry_budget
        self._seed = seed
        self._tokens = float(concurrency)
        self._updated = asyncio.get_running_loop().time()
        self._active = 0
        self._successes = 0
        self._condition = asyncio.Condition()

    def _refill(self) -> None:
        now = asyncio.get_running_loop().time()
        self._tokens = min(self._limit, self._tokens + (now - self._updated) * self._limit * 10)
        self._updated = now

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        async with self._condition:
            while True:
                self._refill()
                if self._active >= self._limit:
                    await self._condition.wait()
                elif self._tokens < 1:
                    try:
                        await asyncio.wait_for(
                            self._condition.wait(), (1 - self._tokens) / (self._limit * 10),
                        )
                    except TimeoutError:
                        continue
                else:
                    self._active += 1
                    self._tokens -= 1
                    break
        try:
            yield
        finally:
            async with self._condition:
                self._active -= 1
                self._condition.notify_all()

    def backoff(self, key: str, attempt: int) -> float:
        digest = hashlib.sha256(f"{self._seed}:{key}:{attempt}".encode()).digest()
        fraction = int.from_bytes(digest[:4], "big") / 2**32
        return min(2.0, 0.1 * 2 ** min(attempt - 1, 5)) * (0.5 + 0.5 * fraction)

    async def request(
        self, operation: Callable[[], Awaitable[httpx.Response]], *, key: str,
    ) -> httpx.Response:
        attempt = 0
        while True:
            try:
                response = await operation()
                response.raise_for_status()
            except (httpx.HTTPStatusError, httpx.ConnectError, httpx.ConnectTimeout) as exc:
                # A refused connection or an explicit rate rejection can be
                # retried without replaying a completed agent turn.
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code != 429:
                    raise
                async with self._condition:
                    self._refill()
                    self._limit = max(1, self._limit // 2)
                    self._tokens = min(self._tokens, self._limit)
                    self._successes = 0
                    self._condition.notify_all()
                    if self._budget == 0:
                        raise
                    self._budget -= 1
                attempt += 1
                delay = self.backoff(key, attempt)
                if isinstance(exc, httpx.HTTPStatusError):
                    retry_after = exc.response.headers.get("Retry-After", "")
                    if retry_after.isascii() and retry_after.isdecimal() and len(retry_after) <= 10:
                        delay = max(delay, int(retry_after))
                await asyncio.sleep(delay)
            else:
                async with self._condition:
                    self._successes += 1
                    if self._successes >= self._limit and self._limit < self._maximum:
                        self._refill()
                        self._limit += 1
                        self._successes = 0
                        self._condition.notify_all()
                return response
