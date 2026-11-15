"""Debounce + dead-man safety between raw predictions and the command we send.

* A new command is only adopted after the same prediction wins N consecutive
  windows, each with confidence >= threshold. Anything else holds the previous
  command (a low-confidence or empty window breaks the streak).
* Dead-man: if the source produces no fresh data for ``deadman_s``, output S,
  and require a full fresh streak before moving again.
* A forced prediction (the EEG stop gate: the driver is moving) is adopted at once.
"""
from __future__ import annotations

from typing import Optional

import commands
import config


class Smoother:
    def __init__(self, n_windows: int = config.SMOOTH_WINDOWS,
                 threshold: float = config.CONFIDENCE_THRESHOLD,
                 deadman_s: float = config.DEADMAN_S) -> None:
        if n_windows < 1:
            raise ValueError("n_windows must be >= 1")
        self.n_windows = n_windows
        self.threshold = threshold
        self.deadman_s = deadman_s
        self.reset()

    def reset(self) -> None:
        self.command: str = commands.STOP
        self.candidate: Optional[str] = None
        self.streak = 0
        self.last_fresh: Optional[float] = None
        self.deadman = True  # no data yet

    def update(self, command: Optional[str], confidence: float, timestamp: float, force: bool = False) -> str:
        """Feed one fresh window's prediction. Returns the (possibly unchanged) command. ``force``: adopt
        it now, without the streak (it still starts one)."""
        self.last_fresh = timestamp if self.last_fresh is None else max(self.last_fresh, timestamp)
        if command is None or (confidence < self.threshold and not force):
            self.candidate, self.streak = None, 0
            return self.command
        commands.validate(command)
        if command == self.candidate:
            self.streak += 1
        else:
            self.candidate, self.streak = command, 1
        if self.streak >= self.n_windows or force:
            self.command = command
        return self.command

    def output(self, now: float) -> str:
        """The command to send right now, applying the dead-man timeout."""
        if self.last_fresh is None or now - self.last_fresh > self.deadman_s:
            self.deadman = True
            self.command = commands.STOP
            self.candidate, self.streak = None, 0
        else:
            self.deadman = False
        return self.command
