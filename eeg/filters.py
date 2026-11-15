"""Streaming filters and spectra."""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import signal

_trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy 2 renamed trapz


def band_sos(fs: float, band: tuple[float, float], notch_hz: Optional[float] = None, order: int = 4) -> np.ndarray:
    band = (band[0], min(band[1], 0.45 * fs))        # e.g. the 45-100 Hz muscle band at a low sample rate
    sos = signal.butter(order, band, btype="bandpass", fs=fs, output="sos")
    if notch_hz and notch_hz < fs / 2:
        b, a = signal.iirnotch(notch_hz, Q=30.0, fs=fs)
        sos = np.vstack([sos, signal.tf2sos(b, a)])
    return sos


class StreamFilter:
    """Causal band-pass (+ mains notch) applied chunk by chunk with its state carried over, so the
    live stream and an offline replay of the same samples give identical output. Starts in steady
    state at the first sample, so an electrode's large DC offset doesn't ring."""

    def __init__(self, fs: float, band: tuple[float, float], notch_hz: Optional[float] = None) -> None:
        self.sos = band_sos(fs, band, notch_hz)
        self._zi: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._zi = None

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or len(x) == 0:
            return x.reshape(0, x.shape[-1] if x.ndim == 2 else 0)
        if self._zi is None:
            self._zi = signal.sosfilt_zi(self.sos)[:, :, None] * x[0][None, None, :]
        y, self._zi = signal.sosfilt(self.sos, x, axis=0, zi=self._zi)
        return y


def spectrum(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Power spectral density (µV²/Hz) of a 1-D signal, Hann window over the whole segment."""
    x = np.asarray(x, dtype=float)
    return signal.periodogram(x - x.mean(), fs=fs, window="hann")


def band_power(freqs: np.ndarray, psd: np.ndarray, band: tuple[float, float]) -> float:
    lo, hi = band
    sel = (freqs >= lo) & (freqs <= hi)
    return float(_trapz(psd[sel], freqs[sel])) if np.any(sel) else 0.0
