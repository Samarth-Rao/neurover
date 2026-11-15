"""EEG conditioning shared by every EEG decoder: streaming filters, artifact check, spectrum."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy import signal

_trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy 2 renamed trapz


class StreamFilter:
    """Causal band-pass (+ optional mains notch), applied chunk by chunk with state carried over.

    Filtering the continuous stream (rather than each window) avoids edge effects.
    The phase shift doesn't matter for CCA, whose sine/cosine references absorb phase.
    """

    def __init__(self, fs: float, band: tuple[float, float], notch_hz: Optional[float] = None,
                 order: int = 4) -> None:
        sos = signal.butter(order, band, btype="bandpass", fs=fs, output="sos")
        if notch_hz and notch_hz < fs / 2:
            b, a = signal.iirnotch(notch_hz, Q=30.0, fs=fs)
            sos = np.vstack([sos, signal.tf2sos(b, a)])
        self.sos = sos
        self._zi: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._zi = None

    def restart_columns(self, cols: Sequence[int], x0: np.ndarray) -> None:
        """Steady state at ``x0`` for some channels only (a device came back with a new DC offset)."""
        if self._zi is None or not len(cols):
            return
        cols = list(cols)
        self._zi[:, :, cols] = signal.sosfilt_zi(self.sos)[:, :, None] * np.asarray(x0, dtype=float)[None, None, :]

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or len(x) == 0:
            return x.reshape(0, x.shape[-1] if x.ndim == 2 else 0)
        if self._zi is None:
            # Start in steady state for the first sample so a big electrode DC offset doesn't ring.
            self._zi = signal.sosfilt_zi(self.sos)[:, :, None] * x[0][None, None, :]
        y, self._zi = signal.sosfilt(self.sos, x, axis=0, zi=self._zi)
        return y


def is_artifact(window_uv: np.ndarray, limit_uv: float) -> bool:
    """True if any sample in the (band-passed) window exceeds the amplitude limit."""
    return bool(np.max(np.abs(window_uv)) > limit_uv) if window_uv.size else False


def spectrum(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Power spectral density (µV²/Hz) of a 1-D signal, Hann window over the whole segment."""
    x = np.asarray(x, dtype=float)
    return signal.periodogram(x - x.mean(), fs=fs, window="hann")


def band_power(freqs: np.ndarray, psd: np.ndarray, band: tuple[float, float]) -> float:
    lo, hi = band
    sel = (freqs >= lo) & (freqs <= hi)
    return float(_trapz(psd[sel], freqs[sel])) if np.any(sel) else 0.0


def channel_permutation(sdk_channels: Sequence[str], channel_map: dict[str, str],
                        display_order: Sequence[str]) -> tuple[list[int], list[str]]:
    """Columns to take from SDK-ordered data, and the physical position of each, in display order."""
    missing = [c for c in sdk_channels if c not in channel_map]
    if missing:
        raise ValueError(f"EEG_CHANNEL_MAP has no entry for SDK channel(s) {missing}")
    positions = [channel_map[c] for c in sdk_channels]
    if len(set(positions)) != len(positions):
        raise ValueError(f"EEG_CHANNEL_MAP maps two channels to the same position: {positions}")
    rank = {p: i for i, p in enumerate(display_order)}
    order = sorted(range(len(positions)), key=lambda i: (rank.get(positions[i], len(rank)), i))
    return order, [positions[i] for i in order]
