from __future__ import annotations

import time


class SystemClock:
    def now_ms(self) -> int:
        return int(time.time() * 1000)


class FixedClock:
    def __init__(self, now_ms: int) -> None:
        self.t = now_ms

    def now_ms(self) -> int:
        return self.t
