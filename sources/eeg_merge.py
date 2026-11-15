"""Merge EEG from several Bluetooth devices onto one timeline.

Each BrainBit streams at 250 Hz on its own crystal, in its own bursts. Their samples are
timestamped against our clock (brainbit_worker.py), then resampled here onto one shared
250 Hz grid by linear interpolation, ``lag_s`` behind real time so every connected
device's packets for that moment have arrived.

Where a device has no data (not connected yet, Bluetooth drop), its channels hold their
last value and ``present`` is False for those samples; the EEG pipeline then leaves that
device out of decoding until it is back. If no device has data, nothing is produced, so
the dead-man stops the rover.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


class StreamMerger:
    def __init__(self, widths: Sequence[int], fs: float, lag_s: float = 0.3, max_gap_s: float = 0.1,
                 keep_s: float = 2.0) -> None:
        self.widths = [int(w) for w in widths]
        self.fs = float(fs)
        self.lag_s = lag_s
        self.max_gap_s = max_gap_s
        self.keep_s = keep_s
        self._t = [np.zeros(0) for _ in self.widths]
        self._x = [np.zeros((0, w)) for w in self.widths]
        self._hold: list[Optional[np.ndarray]] = [None for _ in self.widths]
        self._next: Optional[float] = None

    def push(self, k: int, t: np.ndarray, x: np.ndarray) -> None:
        t = np.asarray(t, dtype=float)
        x = np.asarray(x, dtype=float).reshape(len(t), self.widths[k])
        if len(t) == 0:
            return
        if len(self._t[k]):  # keep timestamps strictly increasing
            keep = t > self._t[k][-1]
            t, x = t[keep], x[keep]
        self._t[k] = np.concatenate([self._t[k], t])
        self._x[k] = np.vstack([self._x[k], x])

    def latest(self, k: int) -> Optional[float]:
        return float(self._t[k][-1]) if len(self._t[k]) else None

    def pull(self, now: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(t[m], x[m, sum(widths)], present[m, devices]) for grid times up to ``now - lag_s``."""
        total = sum(self.widths)
        empty = (np.zeros(0), np.zeros((0, total)), np.zeros((0, len(self.widths)), dtype=bool))
        lasts = [tb[-1] for tb in self._t if len(tb)]
        if not lasts:
            return empty
        dt = 1.0 / self.fs
        if self._next is None:
            self._next = min(tb[0] for tb in self._t if len(tb))
        # Nothing at all around the next grid time (every device was away): skip ahead to the data.
        upcoming = [tb[np.searchsorted(tb, self._next)] for tb in self._t if len(tb) and tb[-1] >= self._next]
        if not upcoming:
            return empty
        first = min(upcoming)
        if first - self._next > self.max_gap_s:
            self._next += np.floor((first - self._next) / dt) * dt
        end = min(now - self.lag_s, max(lasts))
        m = int(np.floor((end - self._next) / dt + 1e-9)) + 1
        if m <= 0:
            return empty
        g = self._next + np.arange(m) * dt
        out = np.zeros((m, total))
        present = np.zeros((m, len(self.widths)), dtype=bool)
        col = 0
        for k, w in enumerate(self.widths):
            out[:, col:col + w], present[:, k] = self._resample(k, g)
            col += w
        self._next = float(g[-1] + dt)
        for k in range(len(self.widths)):  # keep a little history for interpolation brackets
            keep = self._t[k] >= self._next - self.keep_s
            if not keep.all():
                if not keep.any():  # always keep the newest sample
                    keep[-1] = True
                self._t[k], self._x[k] = self._t[k][keep], self._x[k][keep]
        return g, out, present

    def _resample(self, k: int, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tb, xb, w = self._t[k], self._x[k], self.widths[k]
        m = len(g)
        hold = self._hold[k] if self._hold[k] is not None else np.zeros(w)
        if len(tb) == 0:
            return np.tile(hold, (m, 1)), np.zeros(m, dtype=bool)
        i = np.searchsorted(tb, g)                       # first sample at or after each grid time
        hi = np.clip(i, 0, len(tb) - 1)
        lo = np.clip(i - 1, 0, len(tb) - 1)
        exact = (i < len(tb)) & (tb[hi] == g)
        bracket = (i > 0) & (i < len(tb)) & (tb[hi] - tb[lo] <= self.max_gap_s)
        ok = exact | bracket
        vals = np.column_stack([np.interp(g, tb, xb[:, c]) for c in range(w)])
        if not ok.all():  # gaps: hold the last good value (no step for the filters to ring on)
            last_ok = np.maximum.accumulate(np.where(ok, np.arange(m), -1))
            filled = np.where((last_ok >= 0)[:, None], vals[np.maximum(last_ok, 0)], hold[None, :])
            vals = np.where(ok[:, None], vals, filled)
        if ok.any():
            self._hold[k] = vals[np.flatnonzero(ok)[-1]].copy()
        return vals, ok
