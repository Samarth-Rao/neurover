"""Keyboard source: WASD / arrow keys held = F/L/B/R, nothing held = S.

Bypasses the classifier: it emits command letters directly with confidence 1.
It still goes through the smoother like every other source, and it emits one
prediction per hop so the debounce timing matches the EMG pipeline.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import commands
import config

from .base import Prediction, Source, SourceStatus

DIRECTION_TO_COMMAND = {
    "up": commands.FORWARD,
    "down": commands.BACK,
    "left": commands.LEFT,
    "right": commands.RIGHT,
}


class KeyboardSource(Source):
    name = "keyboard"
    classes = commands.ALL

    def __init__(self, period_s: float = config.KEYBOARD_PERIOD_S,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._period = period_s
        self._clock = clock
        self._order: list[str] = []  # held directions, oldest first; the newest one wins
        self._running = False
        self._last: Optional[Prediction] = None

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def status(self) -> SourceStatus:
        return SourceStatus(self._running, "WASD / arrow keys" if self._running else "stopped")

    def on_input(self, held: frozenset[str]) -> None:
        held = {d for d in held if d in DIRECTION_TO_COMMAND}
        kept = [d for d in self._order if d in held]
        self._order = kept + sorted(held.difference(kept))

    def current_command(self) -> str:
        return DIRECTION_TO_COMMAND[self._order[-1]] if self._order else commands.STOP

    def latest(self) -> Optional[Prediction]:
        if not self._running:
            return self._last
        now = self._clock()
        if self._last is None or now - self._last.timestamp >= self._period:
            cmd = self.current_command()
            probs = {c: (1.0 if c == cmd else 0.0) for c in commands.ALL}
            self._last = Prediction(label=cmd, confidence=1.0, emg=None, timestamp=now, probs=probs)
        return self._last
