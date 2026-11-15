"""Per-electrode numbers and spectra, for the live view and for comparing recordings.

Everything is computed from the raw µV samples of one electrode:
    time domain   1-40 Hz + mains notch, zero phase, per stretch of continuous data (Bluetooth gaps
                  split the stream), then 1 s epochs: typical RMS, typical peak, artifact share
    spectrum      Welch PSD (2 s Hann segments = 0.5 Hz bins, as long as a typical decoder window; 50% overlap, mean) over every continuous stretch,
                  leaving out segments with an artifact (|1-40 Hz| > EEG_ARTIFACT_UV, as the rover's
                  decoder does) unless every segment has one. Band powers, mains and SSVEP SNRs come
                  from it; "Artifact time" says how much of the recording artifacts cover.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import signal

import config
from eeg import ssvep
from eeg.filters import band_power

MAX_GAP_S = 0.1        # longer than this between samples = a gap (split the stream there)
SPECTRUM_HZ = (1.0, 45.0)
TARGETS = ssvep.target_frequencies(config.SSVEP_FRAMES, config.SSVEP_REFRESH_HZ)   # label -> Hz
TARGET_ORDER = tuple(l for l in ("up", "down", "left", "right") if l in TARGETS)


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    unit: str
    fmt: str
    help: str


def _snr_metric(label: str) -> Metric:
    f = TARGETS[label]
    return Metric(f"snr_{label}", f"SNR {f:.3g} Hz", "×", "{:.2f}",
                  f"power at {f:.2f} Hz over the mean of its neighbours "
                  f"({config.SSVEP_SNR_NEIGHBOURS_HZ[0]:g}-{config.SSVEP_SNR_NEIGHBOURS_HZ[1]:g} Hz away); "
                  "~1 = nothing there. Record while staring at that box.")


METRICS: tuple[Metric, ...] = (
    Metric("rms", "RMS 1–40 Hz", "µV", "{:.1f}", "typical RMS of the 1-40 Hz signal (median over 1 s epochs)"),
    Metric("peak", "Typical peak", "µV", "{:.0f}",
           f"median over 1 s epochs of the 99th-percentile |1-40 Hz|; the rover app calls "
           f">{config.EEG_QUALITY_WARN_UV:g} MARGINAL and >{config.EEG_QUALITY_MAX_UV:g} NOISY"),
    Metric("line", f"{config.EEG_NOTCH_HZ:g} Hz mains", "µV", "{:.1f}",
           f"RMS of the narrow peak at {config.EEG_NOTCH_HZ:g} Hz above its neighbours (raw, before the notch): "
           "mains pickup, usually a loose or dry contact"),
    Metric("alpha", "Alpha 8–12 Hz", "µV", "{:.1f}", "RMS in 8-12 Hz"),
    Metric("alpha_share", "Alpha share", "%", "{:.0f}", "8-12 Hz power as a share of 4-30 Hz power"),
    *(_snr_metric(label) for label in TARGET_ORDER),
    Metric("artifact", "Artifact time", "%", "{:.0f}",
           f"1 s epochs with any |1-40 Hz| sample over {config.EEG_ARTIFACT_UV:g} µV (blinks, movement, pops)"),
    Metric("dc", "DC offset", "mV", "{:.1f}", "median raw value: drifts as dry electrodes settle; huge = saturated"),
    Metric("dropout", "Missing samples", "%", "{:.1f}", "samples lost to Bluetooth, vs the nominal rate"),
    Metric("resist", "Resistance", "kΩ", "{:.0f}", "last contact measurement (R); good < 1000 kΩ"),
)
METRIC_BY_KEY = {m.key: m for m in METRICS}

_SOS: dict[float, np.ndarray] = {}


def _sos(fs: float) -> np.ndarray:
    if fs not in _SOS:
        sos = signal.butter(4, config.EEG_DISPLAY_BAND, btype="bandpass", fs=fs, output="sos")
        if config.EEG_NOTCH_HZ and config.EEG_NOTCH_HZ < fs / 2:
            b, a = signal.iirnotch(config.EEG_NOTCH_HZ, Q=30.0, fs=fs)
            sos = np.vstack([sos, signal.tf2sos(b, a)])
        _SOS[fs] = sos
    return _SOS[fs]


def runs(t: np.ndarray, max_gap_s: float = MAX_GAP_S) -> list[tuple[int, int]]:
    """[start, stop) index ranges of continuous data."""
    if len(t) == 0:
        return []
    edges = [0, *(np.flatnonzero(np.diff(t) > max_gap_s) + 1).tolist(), len(t)]
    return [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def bandpass(x: np.ndarray, fs: float) -> Optional[np.ndarray]:
    """Zero-phase 1-40 Hz + notch of one continuous stretch (None if too short to filter)."""
    sos = _sos(fs)
    if len(x) < max(int(0.5 * fs), 6 * len(sos) + 1):
        return None
    y = signal.detrend(np.asarray(x, dtype=float), type="linear")
    return signal.sosfiltfilt(sos, y, padlen=min(len(y) - 1, int(fs)))


def analyse(t: np.ndarray, x: np.ndarray, fs: float, *, resistance: Optional[float] = None,
            duration_s: Optional[float] = None, nperseg_s: float = 2.0) -> dict:
    """Every metric in METRICS (NaN where there isn't enough data), plus "freqs"/"psd" (µV²/Hz)
    and "n" (samples). ``t``: seconds, ``x``: one electrode in µV."""
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    out = {m.key: math.nan for m in METRICS}
    out.update(n=len(x), freqs=None, psd=None, saturated=0.0)
    if resistance is not None:
        out["resist"] = resistance / 1e3
    if duration_s and duration_s > 0:
        out["dropout"] = max(0.0, 100.0 * (1.0 - len(x) / (duration_s * fs)))
    if len(x) < 2:
        return out
    out["dc"] = float(np.median(x)) / 1e3
    out["saturated"] = float(np.mean(np.abs(x) >= config.EEG_RAIL_UV))

    ep = max(2, int(round(fs)))
    rms, peak, worst = [], [], []
    spans = runs(t)
    filtered: dict[tuple[int, int], np.ndarray] = {}
    for a, b in spans:
        y = bandpass(x[a:b], fs)
        if y is None:
            continue
        filtered[(a, b)] = y
        k = len(y) // ep
        epochs = y[:k * ep].reshape(k, ep) if k else y[None, :]
        rms += np.sqrt(np.mean(epochs ** 2, axis=1)).tolist()
        peak += np.percentile(np.abs(epochs), 99, axis=1).tolist()
        worst += np.max(np.abs(epochs), axis=1).tolist()
    if rms:
        out["rms"] = float(np.median(rms))
        out["peak"] = float(np.median(peak))
        out["artifact"] = 100.0 * float(np.mean(np.asarray(worst) > config.EEG_ARTIFACT_UV))

    freqs, psd, used, total = _psd(x, spans, filtered, fs, nperseg_s)
    if freqs is None:
        return out
    out["freqs"], out["psd"], out["segments"] = freqs, psd, (used, total)
    if config.EEG_NOTCH_HZ:
        out["line"] = line_noise(freqs, psd, config.EEG_NOTCH_HZ)
    out["alpha"] = math.sqrt(band_power(freqs, psd, config.ALPHA_BAND))
    total = band_power(freqs, psd, config.ALPHA_TOTAL_BAND)
    if total > 0:
        out["alpha_share"] = 100.0 * band_power(freqs, psd, config.ALPHA_BAND) / total
    for label in TARGET_ORDER:
        out[f"snr_{label}"] = snr(freqs, psd, TARGETS[label])
    return out


def _psd(x: np.ndarray, spans: list[tuple[int, int]], filtered: dict, fs: float, nperseg_s: float):
    """(freqs, psd, segments used, segments total): Welch over every continuous stretch, without the
    segments that contain an artifact (unless all of them do)."""
    nper = int(nperseg_s * fs)
    long = [(a, b) for a, b in spans if b - a >= nper]
    if not long:                       # no stretch that long: the longest one, if it's >= 1 s
        a, b = max(spans, key=lambda s: s[1] - s[0])
        if b - a < int(fs):
            return None, None, 0, 0
        nper, long = b - a, [(a, b)]
    step = nper - nper // 2
    cols, bad, freqs = [], [], None
    for a, b in long:
        seg = x[a:b] - np.median(x[a:b])
        freqs, _, sxx = signal.spectrogram(seg, fs=fs, window="hann", nperseg=nper, noverlap=nper // 2,
                                           detrend="linear", scaling="density", mode="psd")
        y = filtered.get((a, b))
        for k in range(sxx.shape[1]):
            cols.append(sxx[:, k])
            bad.append(y is not None and float(np.max(np.abs(y[k * step:k * step + nper]))) > config.EEG_ARTIFACT_UV)
    use = ~np.asarray(bad)
    if not use.any():
        use[:] = True
    return freqs, np.asarray(cols)[use].mean(axis=0), int(use.sum()), len(cols)


def line_noise(freqs: np.ndarray, psd: np.ndarray, f0: float) -> float:
    """RMS (µV) of the narrow peak at ``f0`` above the median level 2-5 Hz either side, so broadband
    muscle noise doesn't count as mains."""
    d = np.abs(freqs - f0)
    peak, side = d <= 1.0, (d >= 2.0) & (d <= 5.0)
    if not peak.any() or not side.any() or len(freqs) < 2:
        return math.nan
    excess = float(np.sum(psd[peak] - np.median(psd[side]))) * float(freqs[1] - freqs[0])
    return math.sqrt(max(0.0, excess))


def snr(freqs: np.ndarray, psd: np.ndarray, f: float) -> float:
    """Power in the bin nearest ``f`` over the mean power of its neighbours."""
    lo, hi = config.SSVEP_SNR_NEIGHBOURS_HZ
    i = int(np.argmin(np.abs(freqs - f)))
    d = np.abs(freqs - f)
    nb = (d >= lo) & (d <= hi)
    if not nb.any():
        return math.nan
    ref = float(np.mean(psd[nb]))
    return float(psd[i]) / ref if ref > 0 else math.nan


def status(stats: dict) -> tuple[str, str]:
    """("OK" | "MARGINAL" | "NOISY" | "FLAT" | "SATURATED" | "NO DATA", severity good|warn|bad|none)."""
    if not stats or stats.get("n", 0) < 2:
        return "NO DATA", "none"
    if stats.get("saturated", 0.0) > 0.01:
        return "SATURATED", "bad"
    peak = stats.get("peak", math.nan)
    if math.isnan(peak):
        return "NO DATA", "none"
    if peak < config.EEG_QUALITY_MIN_UV:
        return "FLAT", "bad"
    if peak > config.EEG_QUALITY_MAX_UV:
        return "NOISY", "bad"
    if peak > config.EEG_QUALITY_WARN_UV:
        return "MARGINAL", "warn"
    return "OK", "good"


def format_value(metric: Metric, value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    if math.isinf(value):
        return "open"
    return metric.fmt.format(value)


def analyse_recording(rec: dict) -> dict[str, dict]:
    """position -> analyse() of that electrode (plus "device"/"label"/"sdk")."""
    from eeg.recording import nodes

    fs = float(rec["meta"]["fs"])
    dur = rec["meta"].get("duration_s")
    out = {}
    for nd in nodes(rec):
        st = analyse(nd["t"], nd["x"], fs, resistance=nd["resistance"], duration_s=dur)
        st.update(device=nd["device"], label=nd["label"], sdk=nd["sdk"])
        out[nd["position"]] = st
    return out
