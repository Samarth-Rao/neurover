"""Mock EMG source: a virtual arm producing Myo-like PREPROCESSED EMG, no hardware needed.

Imitates what the real armband sends in pyomyo's PREPROCESSED mode: 8 rectified,
low-pass-filtered amplitudes at 50 Hz. Each gesture has its own noisy activation
pattern across the channels, and the virtual muscles ramp smoothly between them.
Arrow keys choose the gesture (none held = rest).
"""
from __future__ import annotations

import math
import time
from typing import Callable, Optional

import numpy as np

import config

from .base import SourceStatus
from .signal_source import SignalSource

_ARROWS = {"up": "↑", "down": "↓", "left": "←", "right": "→"}


class MockArm:
    """Deterministic (given a seed) generator of 8-channel EMG amplitudes."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)
        self.patterns = {g: np.asarray(p, dtype=float) for g, p in config.MOCK_PATTERNS.items()}
        self.channels = len(next(iter(self.patterns.values())))
        self.gesture = "rest"
        self.activation = self.patterns["rest"].copy()
        self.noise = np.zeros(self.channels)
        self.effort = 1.0

    def step(self, dt: float) -> np.ndarray:
        target = self.patterns[self.gesture]
        self.activation += (target - self.activation) * (1.0 - math.exp(-dt / config.MOCK_TRANSITION_S))
        # Overall effort wanders slowly, as a real arm's does.
        drift = config.MOCK_EFFORT_DRIFT
        self.effort += (1.0 - self.effort) * 0.02 + self.rng.normal(0.0, drift * 0.04)
        self.effort = float(np.clip(self.effort, 1.0 - drift, 1.0 + drift))
        # AR(1) noise: the Myo's preprocessed stream is low-pass filtered, so noise is correlated.
        rho = 0.7
        self.noise = rho * self.noise + self.rng.normal(0.0, config.MOCK_NOISE, self.channels) * math.sqrt(1 - rho**2)
        amp = self.activation * self.effort * (1.0 + self.noise)
        value = (config.MOCK_BASELINE + config.MOCK_FULL_SCALE * amp
                 + self.rng.normal(0.0, config.MOCK_FLOOR_NOISE, self.channels))
        return np.clip(value, 0.0, None)


class MockEmgSource(SignalSource):
    name = "mock"
    #: Shown next to each gesture in the training panel.
    gesture_hints = {g: _ARROWS[d] for d, g in config.MOCK_ARROW_GESTURES.items()}

    def __init__(self, seed: Optional[int] = None, *, clock: Callable[[], float] = time.monotonic,
                 rate_hz: float = config.MOCK_SAMPLE_HZ, **kwargs) -> None:
        self.arm = MockArm(seed)
        super().__init__(self.arm.channels, "mock", clock=clock, **kwargs)
        self.period = 1.0 / rate_hz
        self._t: Optional[float] = None
        self._running = False
        self._order: list[str] = []

    def start(self) -> None:
        self._running = True
        self._t = self.clock()

    def stop(self) -> None:
        self._running = False

    def status(self) -> SourceStatus:
        if not self._running:
            return SourceStatus(False, "stopped")
        key = next((_ARROWS[d] for d, g in config.MOCK_ARROW_GESTURES.items() if g == self.arm.gesture), "")
        name = config.GESTURE_NAMES[self.arm.gesture]
        return SourceStatus(True, f"virtual arm: {name} {key}".rstrip())

    def on_input(self, held: frozenset[str]) -> None:
        held = {d for d in held if d in config.MOCK_ARROW_GESTURES}
        kept = [d for d in self._order if d in held]
        self._order = kept + sorted(held.difference(kept))
        self.arm.gesture = config.MOCK_ARROW_GESTURES[self._order[-1]] if self._order else "rest"

    def _read_samples(self):
        if not self._running or self._t is None:
            return []
        now = self.clock()
        if now - self._t > 1.0:  # after a long stall, don't generate a huge backlog
            self._t = now - 1.0
        out = []
        while self._t + self.period <= now:
            self._t += self.period
            out.append((self._t, self.arm.step(self.period)))
        return out
