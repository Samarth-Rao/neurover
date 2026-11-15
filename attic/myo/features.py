"""Sliding windows + per-channel features. Generic over channel count.

EMG uses mean absolute value per channel. An EEG source can register a
band-power feature here later and reuse everything else unchanged.
"""
from __future__ import annotations

from collections import deque
from typing import Callable, Iterable, Sequence

import numpy as np


_EPS = 1e-6  # seconds; absorbs float noise in sample timestamps at window edges


class Windower:
    """Turns a timestamped sample stream into overlapping windows.

    A window ending at time ``e`` holds every sample with ``e - window_s < t <= e``.
    Windows end every ``hop_s``. Timestamps (not sample counts) drive it, so it
    works the same at 50 Hz (Myo PREPROCESSED) or 200 Hz, and with bursty delivery.
    """

    def __init__(self, window_s: float, hop_s: float, min_samples: int = 2) -> None:
        if window_s <= 0 or hop_s <= 0:
            raise ValueError("window_s and hop_s must be positive")
        self.window_s = window_s
        self.hop_s = hop_s
        self.min_samples = min_samples
        self.reset()

    def reset(self) -> None:
        self._buf: deque[tuple[float, np.ndarray]] = deque()
        self._anchor: float | None = None  # end time of window 0
        self._k = 0                        # index of the next window to emit

    @property
    def _next_end(self) -> float:
        # anchor + k * hop (not repeated addition) so float error can't accumulate.
        return self._anchor + self._k * self.hop_s

    def push(self, t: float, sample: Sequence[float]) -> list[tuple[float, np.ndarray]]:
        """Add one sample; return any windows completed by it as (end_time, array[n, channels])."""
        x = np.asarray(sample, dtype=float)
        if self._buf and t < self._buf[-1][0]:
            t = self._buf[-1][0]  # never go backwards in time
        self._buf.append((t, x))
        if self._anchor is None:
            self._anchor, self._k = t + self.window_s, 0
        if t - self._next_end > self.window_s:
            self._anchor, self._k = t, 0  # long gap in the stream: skip the empty windows
        eps = _EPS
        out = []
        while t >= self._next_end - eps:
            end = self._next_end
            start = end - self.window_s
            rows = [s for ts, s in self._buf if start + eps < ts <= end + eps]
            if len(rows) >= self.min_samples:
                out.append((end, np.vstack(rows)))
            self._k += 1
        cutoff = self._next_end - self.window_s + eps
        while self._buf and self._buf[0][0] <= cutoff:
            self._buf.popleft()
        return out


def mean_absolute_value(window: np.ndarray) -> np.ndarray:
    """Per-channel mean |x| over a window of shape (samples, channels)."""
    return np.mean(np.abs(np.asarray(window, dtype=float)), axis=0)


# Feature name -> function(window[samples, channels]) -> vector.
# TODO(eeg): add "band_power" (e.g. Welch alpha/beta power per channel).
FEATURES: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "mav": mean_absolute_value,
}


def extract(window: np.ndarray, names: Iterable[str] = ("mav",)) -> np.ndarray:
    return np.concatenate([FEATURES[n](window) for n in names])
