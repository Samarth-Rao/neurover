"""SSVEP scoring methods and score fusion, for offline comparison (tools/ssvep_benchmark.py)
and, for the winner, the live decoder.

Each method maps one EEG window to 4 scores (one per flicker target):
    cca     standard CCA vs sine/cosine references (no training)
    fbcca   filter-bank CCA: CCA in 3 sub-bands, weighted sum of squared correlations (no training)
    snr     power at f and 2f relative to neighbouring frequencies (no training)
    trca    ensemble task-related component analysis: learns per-target spatial filters and
            templates from calibration, phase-locked to the flicker using the recorded frame times
Fusion (trained on calibration data):
    avg       mean of per-method z-scores
    weighted  z-scores weighted by each method's held-out accuracy
    stacked   logistic regression on all z-scores (out-of-fold features to avoid leakage)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

import config
from pipeline import ssvep
from pipeline.eeg import StreamFilter, spectrum

FBCCA_BANDS_LO = (6.0, 14.0, 22.0)      # sub-band lower edges, upper edge = SSVEP_BAND[1]
FBCCA_WEIGHTS = [(k + 1) ** -1.25 + 0.25 for k in range(len(FBCCA_BANDS_LO))]
BASE_METHODS = ("cca", "fbcca", "snr", "trca")
REST = ssvep.REST


@dataclass
class Prepared:
    fs: float
    positions: list[str]
    t: np.ndarray
    xs: np.ndarray                 # SSVEP band, decode channels
    xd: np.ndarray                 # 1-40 Hz (artifact check)
    bands: list[np.ndarray]        # FBCCA sub-bands
    flip_t: np.ndarray
    flip_frame: np.ndarray
    frames: dict[str, int]
    targets: dict[str, float]
    segments: list[tuple[str, float, float]]
    present: Optional[np.ndarray] = None   # [n] every device of these electrodes had data

    def frame_at(self, t: float) -> float:
        return float(np.interp(t, self.flip_t, self.flip_frame, left=np.nan, right=np.nan))

    def time_of_frame(self, f: float) -> float:
        return float(np.interp(f, self.flip_frame, self.flip_t, left=np.nan, right=np.nan))


def prepare(rec: dict, positions: Sequence[str] = config.EEG_DECODE_POSITIONS) -> Prepared:
    fs = rec["fs"]
    cols = [i for i, p in enumerate(rec["positions"]) if p in positions] or list(range(len(rec["positions"])))
    x = rec["x"][:, cols]
    present = None
    if "present" in rec and "devices" in rec:
        used = {rec["positions"][i] for i in cols}
        devs = [k for k, ps in enumerate(rec["devices"].values()) if used & set(ps)]
        if devs and rec["present"].shape[1] > max(devs):
            present = rec["present"][:, devs].all(axis=1)
    hi = config.SSVEP_BAND[1]
    targets = ssvep.target_frequencies(rec["frames"], config.SSVEP_REFRESH_HZ)
    return Prepared(
        fs=fs, positions=[rec["positions"][i] for i in cols], t=rec["t"],
        xs=StreamFilter(fs, config.SSVEP_BAND, config.EEG_NOTCH_HZ).process(x),
        xd=StreamFilter(fs, config.EEG_DISPLAY_BAND, config.EEG_NOTCH_HZ).process(x),
        bands=[StreamFilter(fs, (lo, hi), config.EEG_NOTCH_HZ).process(x) for lo in FBCCA_BANDS_LO],
        flip_t=rec["flip_t"], flip_frame=rec["flip_frame"].astype(float), frames=dict(rec["frames"]),
        targets=targets, segments=list(rec["segments"]), present=present)


@dataclass
class Window:
    label: str
    seg: int
    i0: int
    t0: float
    t_end: float
    clean: bool
    settled: bool        # far enough into the segment that the window holds only this condition


def windows(p: Prepared, window_s: float, hop_s: float = config.SSVEP_HOP_S, settle_s: float = 0.5) -> list[Window]:
    n, hop = int(round(window_s * p.fs)), int(round(hop_s * p.fs))
    out = []
    for si, (label, a, b) in enumerate(p.segments):
        i_first = int(np.searchsorted(p.t, a))
        i_last = int(np.searchsorted(p.t, b)) - n
        for i0 in range(max(0, i_first - n + hop), i_last + 1, hop):  # windows ENDING inside the segment
            t_end = p.t[min(i0 + n - 1, len(p.t) - 1)]
            if t_end < a:
                continue
            clean = not np.max(np.abs(p.xd[i0:i0 + n])) > config.EEG_ARTIFACT_UV
            if p.present is not None and p.present[i0:i0 + n].mean() < 0.9:
                clean = False   # a device was away
            out.append(Window(label, si, i0, float(p.t[i0]), float(t_end), clean, p.t[i0] >= a + settle_s))
    return out


# -- base methods -------------------------------------------------------------------------------
class _Refs:
    def __init__(self) -> None:
        self.cache: dict[tuple[int, float], list[np.ndarray]] = {}

    def get(self, p: Prepared, n: int) -> list[np.ndarray]:
        key = (n, p.fs)
        if key not in self.cache:
            self.cache[key] = [ssvep.reference_signals(f, n, p.fs, config.SSVEP_HARMONICS) for f in p.targets.values()]
        return self.cache[key]


_REFS = _Refs()


def score_cca(p: Prepared, w: Window, n: int) -> np.ndarray:
    x = p.xs[w.i0:w.i0 + n]
    return np.array([ssvep.cca_max_corr(x, r) for r in _REFS.get(p, n)])


def score_fbcca(p: Prepared, w: Window, n: int) -> np.ndarray:
    total = np.zeros(len(p.targets))
    for wk, band in zip(FBCCA_WEIGHTS, p.bands):
        x = band[w.i0:w.i0 + n]
        total += wk * np.array([ssvep.cca_max_corr(x, r) ** 2 for r in _REFS.get(p, n)])
    return total


def score_snr(p: Prepared, w: Window, n: int) -> np.ndarray:
    x = p.xs[w.i0:w.i0 + n]
    psd = None
    for c in range(x.shape[1]):
        freqs, pc = spectrum(x[:, c], p.fs)
        psd = pc if psd is None else psd + pc
    out = []
    for f in p.targets.values():
        s = 0.0
        for h in (1, 2):
            k = int(np.argmin(np.abs(freqs - h * f)))
            nb = (np.abs(freqs - h * f) > 0.6) & (np.abs(freqs - h * f) < 2.0)
            s += math.log((psd[k] + 1e-12) / (np.mean(psd[nb]) + 1e-12))
        out.append(s)
    return np.array(out)


class Trca:
    """Ensemble TRCA with templates phase-locked to each target's flicker cycle."""

    def __init__(self) -> None:
        self.W: Optional[np.ndarray] = None
        self.templates: dict[str, np.ndarray] = {}
        self.m = 0

    def _aligned(self, p: Prepared, w: Window, label: str, n: int) -> Optional[np.ndarray]:
        f0 = p.frame_at(w.t0)
        if math.isnan(f0):
            return None
        period = p.frames[label]
        onset = math.ceil(f0 / period) * period
        t_on = p.time_of_frame(onset)
        if math.isnan(t_on):
            return None
        i = int(np.searchsorted(p.t, t_on))
        if i + self.m > w.i0 + n or i + self.m > len(p.t):
            return None
        return p.xs[i:i + self.m]

    def fit(self, p: Prepared, wins: Sequence[Window], n: int) -> "Trca":
        self.m = n - int(math.ceil(max(p.frames.values()) / config.SSVEP_REFRESH_HZ * p.fs)) - 2
        filters = []
        for label in p.targets:
            trials = [a for w in wins if w.label == label and w.clean and w.settled
                      for a in [self._aligned(p, w, label, n)] if a is not None]
            if len(trials) < 3:
                raise ValueError(f"not enough clean {label} windows to train TRCA")
            X = np.stack([t - t.mean(axis=0) for t in trials])          # trials, samples, channels
            self.templates[label] = X.mean(axis=0)
            s = X.sum(axis=0)
            S = s.T @ s - np.einsum("tij,tik->jk", X, X)                  # inter-trial covariance
            Q = np.einsum("tij,tik->jk", X, X) + 1e-9 * np.eye(X.shape[2])
            vals, vecs = np.linalg.eig(np.linalg.solve(Q, S))
            filters.append(np.real(vecs[:, int(np.argmax(np.real(vals)))]))
        self.W = np.column_stack(filters)                               # ensemble filter: channels x targets
        return self

    def score(self, p: Prepared, w: Window, n: int) -> np.ndarray:
        out = []
        for label in p.targets:
            x = self._aligned(p, w, label, n)
            if x is None:
                out.append(0.0)
                continue
            a = ((x - x.mean(axis=0)) @ self.W).ravel()
            b = (self.templates[label] @ self.W).ravel()
            out.append(float(np.corrcoef(a, b)[0, 1]))
        return np.array(out)


# -- scoring + fusion -----------------------------------------------------------------------------
@dataclass
class Scored:
    wins: list[Window]
    by_method: dict[str, np.ndarray] = field(default_factory=dict)   # method -> (windows, 4)


def score_all(p: Prepared, wins: Sequence[Window], window_s: float, trca: Optional[Trca]) -> Scored:
    n = int(round(window_s * p.fs))
    sc = Scored(list(wins))
    fns = {"cca": score_cca, "fbcca": score_fbcca, "snr": score_snr}
    for name, fn in fns.items():
        sc.by_method[name] = np.array([fn(p, w, n) for w in wins])
    if trca is not None:
        sc.by_method["trca"] = np.array([trca.score(p, w, n) for w in wins])
    return sc


class Fusion:
    """Learned on calibration (training) scores; z-normalises each method, then combines."""

    def __init__(self, methods: Sequence[str]) -> None:
        self.methods = list(methods)
        self.mu: dict[str, np.ndarray] = {}
        self.sd: dict[str, np.ndarray] = {}
        self.weights: dict[str, float] = {}
        self.lr = None

    def z(self, name: str, s: np.ndarray) -> np.ndarray:
        return (s - self.mu[name]) / self.sd[name]

    def fit(self, train: Scored, oof: Optional[Scored] = None, labels: Sequence[str] = ()) -> "Fusion":
        """``oof`` = out-of-fold training scores (for weights and stacking without leakage)."""
        oof = oof or train
        keep = np.array([w.clean for w in train.wins])
        for m in self.methods:
            s = train.by_method[m][keep]
            self.mu[m], self.sd[m] = s.mean(axis=0), s.std(axis=0) + 1e-9
        y = np.array([w.label for w in oof.wins])
        ok = np.array([w.clean and w.settled for w in oof.wins])
        tgt = ok & (y != REST)
        for m in self.methods:
            pred = np.array(labels)[np.argmax(self.z(m, oof.by_method[m][tgt]), axis=1)]
            self.weights[m] = max(0.0, float(np.mean(pred == y[tgt])) - 1.0 / len(labels))
        if sum(self.weights.values()) == 0:
            self.weights = {m: 1.0 for m in self.methods}
        from sklearn.linear_model import LogisticRegression
        X = np.hstack([self.z(m, oof.by_method[m][ok]) for m in self.methods])
        self.lr = LogisticRegression(C=0.3, max_iter=2000).fit(X, y[ok])
        return self

    def combine(self, sc: Scored, how: str, labels: Sequence[str]) -> np.ndarray:
        if how == "avg":
            return np.mean([self.z(m, sc.by_method[m]) for m in self.methods], axis=0)
        if how == "weighted":
            tot = sum(self.weights.values())
            return sum(self.weights[m] / tot * self.z(m, sc.by_method[m]) for m in self.methods)
        if how == "stacked":
            X = np.hstack([self.z(m, sc.by_method[m]) for m in self.methods])
            proba = self.lr.predict_proba(X)
            cls = list(self.lr.classes_)
            return np.column_stack([proba[:, cls.index(l)] if l in cls else np.zeros(len(X)) for l in labels])
        raise ValueError(how)
