"""Mock EEG source: synthetic 4-channel occipital EEG at 250 Hz, no hardware needed.

Signals are synthesised per physical position (O1, Oz, O2, Pz), then handed to
the shared pipeline in the SDK's channel order, exactly like the real headband,
so the channel-mapping config is exercised too. Contents (microvolts):

    pink (1/f) background noise + alpha rhythm (~10 Hz, much stronger with eyes closed)
    + 60 Hz mains + a per-electrode DC offset (both removed by the filters)
    + occasional blinks and jaw clenches (artifacts)
    + while an arrow key is held: a weak SSVEP at that target's frequency and 2nd harmonic.

Keys: arrows = which flicker box the virtual user stares at, E = eyes closed, J = jaw clench.
"""
from __future__ import annotations

import math
import time
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
from scipy import signal

import config
from pipeline import ssvep

from .base import SourceStatus
from .eeg_common import _ARROWS, EegStreamSource

_PINK_B = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
_PINK_A = [1.0, -2.494956002, 2.017265875, -0.522189400]
_DIRECTIONS = ("up", "down", "left", "right")


def _unit_rms(b, a) -> float:
    """RMS of the filter's output for unit white noise (from its impulse response)."""
    impulse = np.zeros(20000)
    impulse[0] = 1.0
    return float(np.sqrt(np.sum(signal.lfilter(b, a, impulse) ** 2)))


class MockEeg:
    def __init__(self, positions: Sequence[str], fs: float, targets: Mapping[str, float],
                 seed: Optional[int] = None, artifacts: bool = True, subject: Optional[int] = None) -> None:
        """``seed``: the noise. ``subject``: the virtual head + electrode placement (per-electrode
        offsets, alpha gains, SSVEP phases); defaults to ``seed``. Same subject + another seed =
        the same person, same session, fresh EEG."""
        self.rng = np.random.default_rng(seed)
        srng = self.rng if subject is None else np.random.default_rng(subject)
        self.fs = fs
        self.positions = tuple(positions)
        self.targets = dict(targets)
        c = len(self.positions)
        self.weight = np.array([config.MOCK_EEG_SPATIAL.get(p, 0.5) for p in self.positions])
        self.artifacts = artifacts
        self.ssvep_uv = config.MOCK_EEG_SSVEP_UV
        self.alpha_open_uv = config.MOCK_EEG_ALPHA_UV   # alpha with eyes open (closed: MOCK_EEG_ALPHA_CLOSED_UV)

        self._pink_zi = np.zeros((len(_PINK_A) - 1, c))
        self._pink_rms = _unit_rms(_PINK_B, _PINK_A)
        # Alpha: white noise through a band-pass, so (like real alpha) its power stays inside ~8-12 Hz.
        self._alpha_sos = signal.butter(4, config.MOCK_EEG_ALPHA_BAND, btype="bandpass", fs=fs, output="sos")
        self._alpha_zi = np.zeros((self._alpha_sos.shape[0], 2))
        impulse = np.zeros(20000)
        impulse[0] = 1.0
        self._alpha_rms = float(np.sqrt(np.sum(signal.sosfilt(self._alpha_sos, impulse) ** 2)))
        self._alpha_level = self.alpha_open_uv
        self._jaw_sos = signal.butter(4, [20.0, min(100.0, fs / 2 - 1)], btype="bandpass", fs=fs, output="sos")
        self._jaw_zi = np.zeros((self._jaw_sos.shape[0], 2, c))

        self.dc = srng.uniform(-1, 1, c) * config.MOCK_EEG_DC_UV
        self.alpha_mix = 1.0 + 0.15 * srng.normal(size=c)  # fixed per-electrode alpha gain
        self.ssvep_phase = srng.uniform(0, 2 * math.pi, c) * 0.15  # small per-site latency spread
        self.mains_phase = srng.uniform(0, 2 * math.pi)
        self.env = {label: 0.0 for label in self.targets}

        self.gaze: Optional[str] = None
        self.eyes_closed = False
        self.jaw = False
        self._manual_jaw = 0.0  # envelope of the held-J clench
        self.n = 0
        self._events: list[tuple[int, int, str, float]] = []  # (start, end, kind, amplitude)
        self._next_blink = self._draw_next(config.MOCK_EEG_BLINK_EVERY_S)
        self._next_jaw = self._draw_next(config.MOCK_EEG_JAW_EVERY_S)

    def _draw_next(self, mean_s: float) -> Optional[int]:
        if not self.artifacts or mean_s <= 0:
            return None
        return self.n + int(self.rng.exponential(mean_s) * self.fs) + int(self.fs)

    def _ramp(self, start: float, target: float, n: int, tau_s: float) -> np.ndarray:
        k = np.arange(1, n + 1)
        return target + (start - target) * np.exp(-k / (tau_s * self.fs))

    def generate(self, n: int) -> np.ndarray:
        if n <= 0:
            return np.zeros((0, len(self.positions)))
        fs, c, w = self.fs, len(self.positions), self.weight
        idx = self.n + np.arange(n)
        t = idx / fs

        pink, self._pink_zi = signal.lfilter(_PINK_B, _PINK_A, self.rng.normal(size=(n, c)), axis=0, zi=self._pink_zi)
        x = pink * (config.MOCK_EEG_NOISE_UV / self._pink_rms)

        a_src, self._alpha_zi = signal.sosfilt(self._alpha_sos, self.rng.normal(size=n), zi=self._alpha_zi)
        if self.eyes_closed:
            target_alpha = config.MOCK_EEG_ALPHA_CLOSED_UV
        else:  # attending a flicker box desynchronises (suppresses) occipital alpha
            target_alpha = self.alpha_open_uv * (config.MOCK_EEG_ALPHA_ATTEND if self.gaze else 1.0)
        level = self._ramp(self._alpha_level, target_alpha, n, 0.4)
        self._alpha_level = float(level[-1])
        alpha = (a_src / self._alpha_rms * level)[:, None]
        x += alpha * (w * self.alpha_mix)[None, :]

        for label, freq in self.targets.items():
            env = self._ramp(self.env[label], 1.0 if self.gaze == label else 0.0, n, 0.3)
            self.env[label] = float(env[-1])
            if env.max() < 1e-3:
                continue
            ph = 2 * math.pi * freq * t[:, None] + self.ssvep_phase[None, :]
            wave = np.sin(ph) + 0.5 * np.sin(2 * ph + 0.7)
            x += self.ssvep_uv * env[:, None] * wave * w[None, :]

        x += config.MOCK_EEG_MAINS_UV * np.sin(2 * math.pi * 60.0 * t + self.mains_phase)[:, None]
        x += self.dc[None, :]
        x += self._artifacts(idx, n)
        self.n += n
        return x

    def _artifacts(self, idx: np.ndarray, n: int) -> np.ndarray:
        fs, c = self.fs, len(self.positions)
        end = self.n + n
        if self._next_blink is not None and self._next_blink < end:
            self._events.append((self._next_blink, self._next_blink + int(0.5 * fs), "blink",
                                 float(self.rng.uniform(20, 160))))
            self._next_blink = self._draw_next(config.MOCK_EEG_BLINK_EVERY_S)
        if self._next_jaw is not None and self._next_jaw < end:
            dur = float(self.rng.uniform(0.6, 1.2))
            self._events.append((self._next_jaw, self._next_jaw + int(dur * fs), "jaw",
                                 float(self.rng.uniform(80, 150))))
            self._next_jaw = self._draw_next(config.MOCK_EEG_JAW_EVERY_S)
        out = np.zeros((n, c))
        manual = self._ramp(self._manual_jaw, 1.0 if self.jaw else 0.0, n, 0.05)
        self._manual_jaw = float(manual[-1])
        jaw_env = manual if manual.max() > 1e-3 else np.zeros(n)
        jaw_amp = 120.0 if manual.max() > 1e-3 else 0.0
        for start, stop, kind, amp in self._events:
            if stop <= self.n or start >= end:
                continue
            k = np.arange(max(start, self.n), min(stop, end))
            if kind == "blink":
                centre = (start + stop) / 2
                bump = amp * np.exp(-(((k - centre) / fs) / 0.07) ** 2)
                out[k - self.n] += bump[:, None] * np.array([1.4 if p == "Pz" else 0.8 for p in self.positions])
            else:
                rise = np.clip(np.minimum(k - start, stop - k) / (0.05 * fs), 0, 1)
                jaw_env[k - self.n] = np.maximum(jaw_env[k - self.n], rise)
                jaw_amp = max(jaw_amp, amp)
        emg, self._jaw_zi = signal.sosfilt(self._jaw_sos, self.rng.normal(size=(n, c)), axis=0, zi=self._jaw_zi)
        if jaw_amp > 0:
            out += emg / (np.std(emg) + 1e-9) * jaw_amp * jaw_env[:, None]
        self._events = [e for e in self._events if e[1] > end]
        return out


class MockEegSource(EegStreamSource):
    name = "mock-eeg"

    def __init__(self, mode: str = "ssvep", seed: Optional[int] = None, artifacts: bool = True, *,
                 ssvep_uv: Optional[float] = None, alpha_uv: Optional[float] = None, subject: Optional[int] = None,
                 clock: Callable[[], float] = time.monotonic, **kwargs) -> None:
        super().__init__(mode, clock=clock, **kwargs)
        # The virtual flicker boxes exist whatever the decoder (alpha mode just never looks at them).
        targets = ssvep.target_frequencies(config.SSVEP_FRAMES, config.SSVEP_REFRESH_HZ)
        self.gen = MockEeg(self.positions, self.fs, targets, seed=seed, artifacts=artifacts, subject=subject)
        if ssvep_uv is not None:
            self.gen.ssvep_uv = ssvep_uv
        if alpha_uv is not None:
            self.gen.alpha_open_uv = self.gen._alpha_level = alpha_uv
        # The generator works by physical position; the headband reports SDK channels.
        self._sdk_from_phys = [self.positions.index(self.channel_map[ch]) for ch in self.sdk_channels]
        self.contact = {p: True for p in self.positions}
        self._order: list[str] = []
        self._t: Optional[float] = None
        self._running = False

    def start(self) -> None:
        self._running = True
        self._t = self.clock()

    def stop(self) -> None:
        self._running = False

    def on_input(self, held: frozenset[str]) -> None:
        dirs = {d for d in held if d in _DIRECTIONS}
        kept = [d for d in self._order if d in dirs]
        self._order = kept + sorted(dirs.difference(kept))
        self.gen.gaze = self._order[-1] if self._order else None
        self.gen.eyes_closed = "eyes" in held
        self.gen.jaw = "jaw" in held

    def status(self) -> SourceStatus:
        if not self._running:
            return SourceStatus(False, "stopped")
        g = self.gen.gaze
        if self.mode == "alpha":
            parts = ["virtual eyes: CLOSED" if self.gen.eyes_closed else "virtual eyes: open"]
        else:
            parts = [f"virtual gaze: {_ARROWS[g]} {self.gen.targets[g]:.1f} Hz" if g else "virtual gaze: centre"]
            if self.gen.eyes_closed:
                parts.append("eyes closed")
        if self.gen.jaw:
            parts.append("jaw clench")
        warn = self.electrode_warning()
        if warn is not None:
            return SourceStatus(False, warn[0], pending=not warn[1])
        return SourceStatus(True, " · ".join(parts))

    def _read_chunk(self) -> tuple[np.ndarray, np.ndarray]:
        empty = (np.zeros(0), np.zeros((0, len(self.sdk_channels))))
        if not self._running or self._t is None:
            return empty
        now = self.clock()
        if now - self._t > 1.0:
            self._t = now - 1.0
        n = int((now - self._t) * self.fs + 1e-9)
        if n <= 0:
            return empty
        t = self._t + np.arange(1, n + 1) / self.fs
        self._t = float(t[-1])
        phys = self.gen.generate(n)
        return t, phys[:, self._sdk_from_phys]
