"""Client-side request-weight limiter (sliding window, conservative)."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable


class WeightLimiter:
    """Blocks until `weight` fits within `budget = limit * safety` over a sliding window.

    Binance counts weight per fixed wall-clock window; a sliding window is strictly more
    conservative. `observe_server_used` folds in the server's own count (from the
    `X-MBX-USED-WEIGHT-1M` header) so other processes sharing the IP are accounted for.
    """

    def __init__(
        self,
        limit: int,
        window_s: float,
        *,
        safety: float = 0.8,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not 0 < safety <= 1:
            raise ValueError("safety must be in (0, 1]")
        self.budget = max(1, int(limit * safety))
        self.window_s = window_s
        self._clock = clock
        self._sleep = sleep
        self._log: deque[tuple[float, int]] = deque()
        self._lock = threading.Lock()

    def _purge(self, now: float) -> None:
        while self._log and self._log[0][0] <= now - self.window_s:
            self._log.popleft()

    def used(self) -> int:
        with self._lock:
            self._purge(self._clock())
            return sum(w for _, w in self._log)

    def acquire(self, weight: int) -> float:
        """Reserve `weight`; returns seconds slept."""
        if weight > self.budget:
            raise ValueError(f"weight {weight} exceeds budget {self.budget}")
        slept = 0.0
        while True:
            with self._lock:
                now = self._clock()
                self._purge(now)
                used = sum(w for _, w in self._log)
                if used + weight <= self.budget:
                    self._log.append((now, weight))
                    return slept
                wait = self._log[0][0] + self.window_s - now + 0.01
            self._sleep(wait)
            slept += wait

    def observe_server_used(self, server_used: int) -> None:
        with self._lock:
            now = self._clock()
            self._purge(now)
            local = sum(w for _, w in self._log)
            if server_used > local:
                self._log.append((now, server_used - local))

    def penalize(self, seconds: float) -> None:
        """After a 429: block the whole budget for `seconds`."""
        with self._lock:
            now = self._clock()
            self._log.clear()
            # An entry that expires `seconds` from now and consumes the full budget.
            self._log.append((now - self.window_s + seconds, self.budget))
