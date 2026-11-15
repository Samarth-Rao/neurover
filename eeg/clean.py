"""Noise handling, kept simple. The rules, and why:

* Each device has its own reference electrode, and that reference's noise (plus head movement) is
  shared by all of the device's channels: 60-97% of the power on the real Flex and headband. Judged on
  the raw referential channels, one noisy reference makes every electrode of that device look bad at
  once, so adding a device just added ways to fail. Electrodes are therefore judged on the signal
  with the device's shared part removed (each electrode minus the median of its device-mates).
* An electrode's level is robust: each window's median absolute deviation, then the median over the
  last few seconds, so blinks and pops don't move it.
* An electrode sits out of decoding only if it's flat, saturated, or clearly noisier than its
  device's other electrodes (or than a sane absolute limit). It has to look that way for a couple of
  seconds to go out, and look fine as long to come back, so it doesn't flicker.
* Nothing ever throws a whole window away. Samples far outside an electrode's typical level are
  down-weighted in the decoder (weight (k/o)^2 beyond k typical levels), so a blink or a reference
  pop shrinks to almost nothing and the rest of the window still counts.
* Movement is different: the driver turning their head (to look at the rover), electrodes shifting, or
  a clenched jaw means they aren't looking at a box, and whatever the boxes score then is noise. So
  MotionCheck watches the newest half second of each device and the decoder stops the rover while it
  fires (eeg/ssvep.py StopGate). A clenched jaw is a deliberate stop, too.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

import config

OK, NOISY, FLAT, SATURATED, UNKNOWN = "ok", "noisy", "flat", "saturated", "unknown"
MOVING, MUSCLE = "moving", "muscle"          # MotionCheck: why the decoder stops


def mad_sd(x: np.ndarray) -> np.ndarray:
    """Robust standard deviation per column (median absolute deviation x 1.4826)."""
    return 1.4826 * np.median(np.abs(x - np.median(x, axis=0)), axis=0)


def shared_part(x: np.ndarray) -> np.ndarray:
    """A device's shared (reference) signal: the median over its electrodes, sample by sample."""
    return np.median(x, axis=1) if x.shape[1] >= 2 else np.zeros(len(x))


def without_shared(x: np.ndarray) -> np.ndarray:
    """Each electrode minus the median of the device's OTHER electrodes. Subtracting the plain mean would
    smear one bad electrode's noise into all the others (it then looks only ~2.5x worse than they do);
    the median of the others ignores it. With two electrodes both get the same (a - b) / 2."""
    c = x.shape[1]
    if c == 1:
        return x
    if c == 2:
        return x - x.mean(axis=1, keepdims=True)
    out = np.empty_like(x)
    for i in range(c):
        out[:, i] = x[:, i] - np.median(np.delete(x, i, axis=1), axis=1)
    return out


class Cleaner:
    """Per device: judges its electrodes window by window, and weights samples for the decoder."""

    def __init__(self, channels: int, history_s: float = config.CLEAN_HISTORY_S,
                 hold_s: float = config.CLEAN_HOLD_S) -> None:
        self.channels = channels
        self.history_s = history_s
        self.hold_s = hold_s
        self._hist: deque = deque()                # (t, level with shared part removed, level as recorded, shared)
        self.status = [UNKNOWN] * channels
        self._pending: list[Optional[tuple[str, float]]] = [None] * channels
        self.level = np.full(channels, np.nan)     # µV, typical, shared part removed (what the dashboard shows)
        self.raw_level = np.full(channels, np.nan)  # µV, typical, as recorded (scale for the sample weights)
        self.shared_level = float("nan")           # µV, typical level of the device's shared (reference) part

    def update(self, t_end: float, x: np.ndarray, raw: Optional[np.ndarray] = None) -> None:
        """Judge the electrodes on one window: ``x`` filtered EEG [n, channels], ``raw`` the same
        samples unfiltered (for saturation)."""
        if len(x) < 10:
            return
        self._hist.append((t_end, mad_sd(without_shared(x)), mad_sd(x), float(mad_sd(shared_part(x)[:, None])[0])))
        while self._hist and t_end - self._hist[0][0] > self.history_s:
            self._hist.popleft()
        self.level = np.median([h[1] for h in self._hist], axis=0)
        self.raw_level = np.median([h[2] for h in self._hist], axis=0)
        self.shared_level = float(np.median([h[3] for h in self._hist]))
        railed = (np.max(np.abs(raw), axis=0) >= config.EEG_RAIL_UV) if raw is not None and len(raw) \
            else np.zeros(self.channels, dtype=bool)
        settled = t_end - self._hist[0][0] >= 1.0      # judge on ~1 s of history at least
        for c in range(self.channels):
            want = self._judge(c, railed[c])
            if self.status[c] == UNKNOWN and settled:
                self.status[c], self._pending[c] = want, None
            elif want == self.status[c]:
                self._pending[c] = None
            elif self._pending[c] is None or self._pending[c][0] != want:
                self._pending[c] = (want, t_end)
            elif t_end - self._pending[c][1] >= self.hold_s:
                self.status[c], self._pending[c] = want, None

    def _judge(self, c: int, railed: bool) -> str:
        if railed:
            return SATURATED
        if self.raw_level[c] < config.CLEAN_FLAT_UV:
            return FLAT
        lv = self.level[c]
        if lv > config.CLEAN_NOISY_MAX_UV:
            return NOISY
        if self.channels >= 3:   # with two electrodes "minus the mean" is the same for both: can't compare
            others = np.median(np.delete(self.level, c))
            if lv > config.CLEAN_NOISY_FLOOR_UV and lv > config.CLEAN_NOISY_FACTOR * others:
                return NOISY
        return OK

    def usable(self) -> list[int]:
        """Electrodes to decode from (not yet judged = usable)."""
        return [c for c in range(self.channels) if self.status[c] in (OK, UNKNOWN)]

    def weights(self, x: np.ndarray, cols: Optional[list[int]] = None) -> np.ndarray:
        """Per-sample weight for the decoder: 1, or (k/o)^2 where a sample is o > k typical levels out on
        any of ``cols``."""
        cols = list(range(self.channels)) if cols is None else cols
        scale = self.raw_level[cols]
        if not len(cols) or not np.all(np.isfinite(scale)):
            return np.ones(len(x))
        o = np.max(np.abs(x[:, cols] - np.median(x[:, cols], axis=0)) / np.maximum(scale, 1e-3), axis=1)
        k = config.CLEAN_SPIKE_K
        return np.where(o <= k, 1.0, (k / np.maximum(o, 1e-9)) ** 2)


class MotionCheck:
    """Per device: is the driver moving (head turning, electrodes shifting) or clenching right now?

    Judged on the newest MOVE_RECENT_S against the device's calm levels, as recorded (movement and
    jaw muscle reach every electrode, mostly through the shared reference):
    MOVING if more than MOVE_SHARE of the samples (6-40 Hz) are beyond CLEAN_SPIKE_K calm levels,
    MUSCLE if the 45-100 Hz level is above MOVE_MUSCLE_FACTOR x calm; either on at least half the
    electrodes in use, so one loose electrode popping doesn't stop the rover (its pops are down-weighted).
    Calm levels come only from undisturbed checks, so a long disturbance can't become the norm, unless it
    goes on for MOVE_RELEARN_S without a break: then the electrodes have changed, and it starts over."""

    def __init__(self, channels: int, calm_s: float = config.MOVE_CALM_S,
                 relearn_s: float = config.MOVE_RELEARN_S) -> None:
        self.channels = channels
        self.calm_s = calm_s
        self.relearn_s = relearn_s
        self._calm: deque = deque()             # (t, decode-band level, muscle-band level), calm checks only
        self._since: Optional[float] = None     # disturbed without a break since
        self.state: Optional[str] = None        # MOVING, MUSCLE or None
        self.share = np.zeros(channels)         # latest: share of samples beyond k calm levels
        self.ratio = np.ones(channels)          # latest: muscle-band level / calm

    def update(self, t_end: float, x: np.ndarray, m: np.ndarray, cols: Optional[list[int]] = None) -> Optional[str]:
        """``x`` the newest samples in the decode band, ``m`` the same samples in the muscle band
        [n, channels]; ``cols`` the electrodes in use. MOVING, MUSCLE, or None (calm, or too little
        calm history yet to tell)."""
        if len(x) < 10 or len(m) != len(x):
            return self.state
        cols = list(range(self.channels)) if cols is None else list(cols)
        lv_x, lv_m = mad_sd(x), mad_sd(m)
        state = None
        if cols and self._calm and t_end - self._calm[0][0] >= 2.0:
            calm_x = np.maximum(np.median([c[1] for c in self._calm], axis=0), 1e-3)
            calm_m = np.maximum(np.median([c[2] for c in self._calm], axis=0), 1e-3)
            self.share = np.mean(np.abs(x - np.median(x, axis=0)) > config.CLEAN_SPIKE_K * calm_x, axis=0)
            self.ratio = lv_m / calm_m
            need = (len(cols) + 1) // 2
            if np.sum(self.ratio[cols] > config.MOVE_MUSCLE_FACTOR) >= need:    # muscle (EMG) first: a clench
                state = MUSCLE                                                   # also shows at 20-40 Hz
            elif np.sum(self.share[cols] > config.MOVE_SHARE) >= need:
                state = MOVING
        if state is None:
            self._since = None
            self._calm.append((t_end, lv_x, lv_m))
            while t_end - self._calm[0][0] > self.calm_s:
                self._calm.popleft()
        elif self._since is None:
            self._since = t_end
        elif t_end - self._since >= self.relearn_s:
            self._calm.clear()
            self._since = None
        self.state = state
        return state
