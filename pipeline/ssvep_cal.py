"""SSVEP calibration: the protocol, a decoder learned from it, and choosing the fastest safe setting.

Protocol (``protocol``): short cued trials (default 5 s) in random order: each box once and the
centre twice per block, 5 blocks, a blink break between blocks. Random order means the data
includes real gaze switches, and the centre (= stop) is sampled across the whole session.

Per window and device, ``FeatureExtractor`` computes (the same code runs live and offline):
    rho[target]            CCA correlation with sin/cos references (fundamental + 2nd harmonic)
    z[target, harmonic]    complex Fourier coefficient of each electrode at that frequency

Decoders:
    cca      mean CCA correlation over devices, z-scored against centre-gaze windows (so one
             threshold works whichever devices are present)
    pattern  learned per device, target and harmonic: the electrodes' covariance at that
             frequency while looking at that box (S) and while not (R). The best spatial filter
             w (top generalised eigenvector of S vs R) gives SNR_k, and each window gets the
             log-likelihood ratio  sum_k |w'z|^2 * snr/(1+snr) - log(1+snr)  of "looking at
             this box" vs "not". Phase-free (no need to know the flicker phase), robust to the
             two Bluetooth devices' clock offset (each device is scored on its own), and it
             adds up evidence over devices and harmonics properly.

Selection (``fit``): leave-one-block-out. For every decoder x window length, the model is fit on
the other blocks and the held-out block is replayed in time order through the same smoother the
rover uses, for a grid of threshold x margin x debounce. For each trial: how long until the
right command (going, or stopping for centre trials), and any wrong drive commands. The setting
with the least time per trial not doing what was asked (latency to go or stop, plus stutter),
whose wrong drive commands stay under SSVEP_CAL_MAX_FALSE and whose driving while looking at the
centre stays under SSVEP_CAL_MAX_REST_DRIVE, wins; the final model is then fit on all blocks.
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
from scipy import linalg

import config
from pipeline.eeg import StreamFilter

REST = "rest"
SKIP = ("ready", "break")
_ARROWS = {"up": "↑", "down": "↓", "left": "←", "right": "→"}
_SHRINK = 0.1        # noise covariance shrinkage towards its diagonal mean
_MIN_SNR = 0.02      # below this a (device, target, harmonic) term contributes nothing


# -- protocol --------------------------------------------------------------------------------------
def protocol(targets: Mapping[str, float], blocks: int = config.SSVEP_CAL_BLOCKS,
             trial_s: float = config.SSVEP_CAL_TRIAL_S, rest_per_block: int = config.SSVEP_CAL_REST_PER_BLOCK,
             break_s: float = config.SSVEP_CAL_BREAK_S, lead_s: float = config.SSVEP_CAL_LEAD_S,
             seed: Optional[int] = None) -> list[tuple[str, float, str]]:
    """Guided calibration steps: (label, seconds, prompt). Labels: targets, REST, "ready", "break"."""
    rng = random.Random(seed)
    segs = [("ready", lead_s, "Get ready: look at the CENTRE (the rover)")]
    prev = REST
    for b in range(blocks):
        trials = list(targets) + [REST] * rest_per_block
        for _ in range(200):  # no box twice in a row, so every trial is a real gaze switch
            rng.shuffle(trials)
            if trials[0] != prev and all(a != c for a, c in zip(trials, trials[1:])):
                break
        for label in trials:
            where = "the CENTRE (the rover)" if label == REST else \
                f"the {_ARROWS.get(label, label)} box ({targets[label]:.1f} Hz)"
            segs.append((label, trial_s, f"Block {b + 1}/{blocks}: look at {where}"))
            prev = label
        if b < blocks - 1:
            segs.append(("break", break_s, "Break: blink and relax your eyes"))
            prev = REST
    return segs


@dataclass
class Trial:
    label: str
    onset: float
    end: float
    block: int
    prev: str            # what the user was asked before (REST at the start of a block)


def trials_from_segments(segments: Sequence[tuple[str, float, float]]) -> list[Trial]:
    """Trials (cued gaze periods) from timed segments; blocks are separated by breaks. Recordings
    from the older protocol (one long segment per condition, no breaks) are split into quarters,
    and each quarter is its own block, so the fitting can still hold data out."""
    real = [s for s in segments if s[0] not in SKIP]
    if any(s[0] == "break" for s in segments) or len(real) > 2 * len({s[0] for s in real}):
        out, block, prev = [], 0, REST
        for label, a, b in segments:
            if label == "break":
                block, prev = block + 1, REST
                continue
            if label in SKIP:
                continue
            out.append(Trial(label, a, b, block, prev))
            prev = label
        return out
    out = []
    for label, a, b in real:
        q = (b - a) / 4
        out += [Trial(label, a + k * q, a + (k + 1) * q, k, label if k else REST) for k in range(4)]
    return sorted(out, key=lambda tr: tr.onset)


# -- which electrodes each device decodes from (live and offline) --------------------------------------
@dataclass
class Layout:
    positions: tuple[str, ...]
    device_names: tuple[str, ...]
    device_of: list[int]
    decode_idx: list[int]
    noise_idx: list[int]

    def columns(self, k: int, unusable: Sequence[int] = ()) -> list[int]:
        cols = [i for i in self.decode_idx if self.device_of[i] == k and i not in unusable]
        if not cols:
            return []
        return cols + [i for i in self.noise_idx if self.device_of[i] == k and i not in unusable]

    def groups(self, xd: np.ndarray, present: Sequence[float], unusable: Sequence[int] = (),
               muscle: Optional[np.ndarray] = None, max_blanked: float = 0.0, fs: float = config.EEG_SAMPLE_HZ
               ) -> tuple[dict[str, list[int]], set[str], dict[str, np.ndarray]]:
        """Device -> columns to decode this window from, the devices an artifact knocked out, and
        device -> keep-mask over the window's samples (only for devices with blanked samples).

        A device needs data for 90% of the window. Samples where a decoding electrode exceeds
        EEG_ARTIFACT_UV, or (``muscle``: [n, channels] flags) any of its electrodes shows a muscle
        burst, are bad; they're blanked with a little padding. If more than ``max_blanked`` of the
        window is bad (0 = any bad sample), the device sits the window out.
        ``xd``: the EEG the artifact check looks at (see config.SSVEP_ARTIFACT_IN_DECODED_BAND)."""
        groups, artifact, masks = {}, set(), {}
        pad = int(round(config.EEG_BLANK_PAD_S * fs))
        for k, name in enumerate(self.device_names):
            if present[k] < 0.9:
                continue
            cols = self.columns(k, unusable)
            if not cols:
                continue
            dec = [i for i in cols if i in self.decode_idx]
            bad = np.any(np.abs(xd[:, dec]) > config.EEG_ARTIFACT_UV, axis=1)
            if muscle is not None:   # electrodes already left out (bad contact) don't get a say
                mine = [i for i in range(len(self.device_of)) if self.device_of[i] == k and i not in unusable]
                if mine:
                    bad |= np.any(muscle[:, mine], axis=1)
            if bad.any():
                if pad:
                    bad = np.convolve(bad.astype(float), np.ones(2 * pad + 1), mode="same") > 0
                if bad.mean() > max_blanked:
                    artifact.add(name)
                    continue
                masks[name] = ~bad
            groups[name] = cols
        return groups, artifact, masks


# -- per-window features ------------------------------------------------------------------------------
def cca_max_corr(X: np.ndarray, Y: np.ndarray) -> float:
    """Largest canonical correlation between the columns of X and Y (QR + SVD)."""
    X = np.asarray(X, dtype=float)
    X = X - X.mean(axis=0)
    X = X[:, X.std(axis=0) > 1e-9]  # a flat (disconnected) channel would make QR meaningless
    if X.shape[1] == 0:
        return 0.0
    Y = Y - Y.mean(axis=0)
    qx, _ = np.linalg.qr(X)
    qy, _ = np.linalg.qr(Y)
    s = np.linalg.svd(qx.T @ qy, compute_uv=False)
    return float(min(1.0, s[0])) if s.size else 0.0


class FeatureExtractor:
    def __init__(self, fs: float, targets: Mapping[str, float], harmonics: int = config.SSVEP_HARMONICS) -> None:
        self.fs = fs
        self.labels = list(targets)
        self.freqs = np.array([targets[l] for l in self.labels])
        self.harmonics = harmonics
        self._cache: dict[int, tuple[list[np.ndarray], np.ndarray]] = {}

    def refs(self, n: int) -> tuple[list[np.ndarray], np.ndarray]:
        hit = self._cache.get(n)
        if hit is None:
            t = np.arange(n) / self.fs
            hz = self.freqs[:, None] * np.arange(1, self.harmonics + 1)[None, :]       # (F, H)
            ph = 2 * np.pi * hz[None, :, :] * t[:, None, None]                          # (n, F, H)
            cca = [np.column_stack([f(ph[:, i, h]) for h in range(self.harmonics) for f in (np.sin, np.cos)])
                   for i in range(len(self.freqs))]
            basis = np.exp(-1j * ph).reshape(n, -1) * (2.0 / n)                         # (n, F*H)
            hit = (cca, basis)
            if len(self._cache) > 8:
                self._cache.clear()
            self._cache[n] = hit
        return hit

    def window(self, xs: np.ndarray, groups: Mapping[str, Sequence[int]], need_cca: bool = True,
               need_z: bool = True, masks: Optional[Mapping[str, np.ndarray]] = None,
               detrend: int = config.SSVEP_DETREND_ORDER) -> dict[str, tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
        """Device -> (rho[F], z[F, H, C]) for one window of SSVEP-band EEG. ``masks``: device ->
        samples to keep (blanked artifacts left out). Slow drift is fitted out (polynomial of order
        ``detrend``) before scoring."""
        n = len(xs)
        cca, basis = self.refs(n)
        out = {}
        for g, cols in groups.items():
            keep = None if masks is None else masks.get(g)
            rows = np.arange(n) if keep is None else np.flatnonzero(keep)
            X = _detrend(xs[rows][:, list(cols)], rows / self.fs, detrend)
            # Patched windows only ever look LESS convincing: fewer samples make chance correlations
            # bigger, so CCA is scaled by sqrt(kept share) (the full-window null scale); the Fourier
            # coefficients keep the full-window normalisation (blanked samples count as zeros).
            keep_share = len(rows) / n
            rho = (np.array([cca_max_corr(X, r[rows]) for r in cca]) * math.sqrt(keep_share)) if need_cca else None
            z = None
            if need_z:
                z = (X.T @ basis[rows]).T.reshape(len(self.freqs), self.harmonics, len(cols))
            out[g] = (rho, z)
        return out


def _detrend(X: np.ndarray, t: np.ndarray, order: int) -> np.ndarray:
    """Remove the mean and (order >= 1) a low-order polynomial trend from each column."""
    if order <= 0 or len(X) <= order + 2:
        return X - X.mean(axis=0)
    u = (t - t.mean()) / (np.ptp(t) / 2 or 1.0)
    P = np.column_stack([u ** k for k in range(order + 1)])
    coef, *_ = np.linalg.lstsq(P, X, rcond=None)
    return X - P @ coef


class MuscleDetector:
    """Streaming muscle-burst flags per electrode: 45-100 Hz level (0.1 s mean |x|) above
    EEG_MUSCLE_FACTOR x that electrode's own slow baseline (geometric mean over ~20 s) and above
    EEG_MUSCLE_MIN_UV. Chunk-invariant (all state carried), so live and offline agree exactly."""

    def __init__(self, fs: float, channels: int) -> None:
        from scipy import signal as sps

        hi = min(config.EEG_MUSCLE_BAND[1], 0.45 * fs)
        self._band = StreamFilter(fs, (config.EEG_MUSCLE_BAND[0], hi), config.EEG_NOTCH_HZ)
        self._box = np.ones(max(1, int(0.1 * fs))) / max(1, int(0.1 * fs))
        self._box_zi: Optional[np.ndarray] = None
        a = math.exp(-1.0 / (config.EEG_MUSCLE_BASELINE_S * fs))
        self._ew = (np.array([1 - a]), np.array([1.0, -a]))
        self._ew_zi: Optional[np.ndarray] = None
        self._warmup = int(0.5 * fs)              # the baseline starts at each electrode's first 0.5 s
        self._warm: list[np.ndarray] = []
        self._sps = sps
        self.channels = channels

    def process(self, x: np.ndarray) -> np.ndarray:
        sps = self._sps
        x = np.asarray(x, dtype=float)
        if len(x) == 0:
            return np.zeros((0, self.channels), dtype=bool)
        hb = np.abs(self._band.process(x))
        if self._box_zi is None:
            self._box_zi = np.zeros((len(self._box) - 1, x.shape[1]))
        env, self._box_zi = sps.lfilter(self._box, 1.0, hb, axis=0, zi=self._box_zi)
        flags = np.zeros(env.shape, dtype=bool)
        start = 0
        if self._ew_zi is None:                   # warming up: no flags, collect the first 0.5 s
            need = self._warmup - sum(len(w) for w in self._warm)
            self._warm.append(env[:need])
            start = min(need, len(env))
            if sum(len(w) for w in self._warm) < self._warmup:
                return flags
            first = np.log(np.maximum(np.median(np.vstack(self._warm), axis=0), 0.5))
            self._ew_zi = sps.lfilter_zi(*self._ew)[:, None] * first[None, :]
            self._warm = []
        rest = env[start:]
        if len(rest):
            base, self._ew_zi = sps.lfilter(*self._ew, np.log(np.maximum(rest, 0.05)), axis=0, zi=self._ew_zi)
            flags[start:] = (rest > config.EEG_MUSCLE_FACTOR * np.exp(base)) & (rest > config.EEG_MUSCLE_MIN_UV)
        return flags


# -- the learned "pattern" decoder ------------------------------------------------------------------
def _cov(z: np.ndarray) -> np.ndarray:
    return (z.T @ z.conj()) / max(1, len(z))


def _shrink(r: np.ndarray) -> np.ndarray:
    c = r.shape[0]
    return (1 - _SHRINK) * r + _SHRINK * (np.real(np.trace(r)) / c) * np.eye(c)


class PatternScorer:
    """Live/offline scoring with fitted (S, R) per (device, target, harmonic), over any electrode subset."""

    def __init__(self, stats: Mapping[tuple[str, int, int], tuple[np.ndarray, np.ndarray]],
                 positions: Mapping[str, Sequence[str]], n_targets: int, harmonics: int) -> None:
        self.stats = dict(stats)
        self.positions = {d: list(p) for d, p in positions.items()}
        self.F, self.H = n_targets, harmonics
        self._filters: dict[tuple, tuple[np.ndarray, float]] = {}

    def _filter(self, dev: str, f: int, h: int, cols: tuple[str, ...]) -> Optional[tuple[np.ndarray, float]]:
        key = (dev, f, h, cols)
        if key not in self._filters:
            st = self.stats.get((dev, f, h))
            if st is None:
                self._filters[key] = None
            else:
                idx = [self.positions[dev].index(p) for p in cols]
                S, R = st[0][np.ix_(idx, idx)], _shrink(st[1][np.ix_(idx, idx)])
                try:
                    vals, vecs = linalg.eigh(S, R)
                    snr = float(vals[-1]) - 1.0
                    self._filters[key] = (vecs[:, -1], snr) if snr > _MIN_SNR else None
                except (linalg.LinAlgError, ValueError):
                    self._filters[key] = None
        return self._filters[key]

    def llr(self, dev: str, cols: Sequence[str], z: np.ndarray) -> np.ndarray:
        """Log-likelihood ratio per target for one device's window (z: [F, H, len(cols)])."""
        cols = tuple(cols)
        if any(p not in self.positions.get(dev, ()) for p in cols):
            keep = [i for i, p in enumerate(cols) if p in self.positions.get(dev, ())]
            cols, z = tuple(cols[i] for i in keep), z[:, :, keep]
        out = np.zeros(self.F)
        if not cols:
            return out
        for f in range(self.F):
            for h in range(self.H):
                filt = self._filter(dev, f, h, cols)
                if filt is None:
                    continue
                w, snr = filt
                t = float(np.abs(np.vdot(w, z[f, h])) ** 2)   # w^H z, with w^H R w = 1
                out[f] += t * snr / (1 + snr) - math.log1p(snr)
        return out


def fit_pattern(feats: "Analysis", train: np.ndarray, label_idx: np.ndarray, n_targets: int) -> PatternScorer:
    stats = {}
    rest = n_targets
    for d, dev in enumerate(feats.devices):
        use = train & feats.used[:, d]
        if not use.any():
            continue
        Z, lab = feats.z[dev][use], label_idx[use]
        for f in range(n_targets):
            sig, at_rest, other = lab == f, lab == rest, (lab != f) & (lab != rest) & (lab >= 0)
            if sig.sum() < 8 or at_rest.sum() < 8:
                continue
            for h in range(feats.harmonics):
                zf = Z[:, f, h, :]
                r_rest = _cov(zf[at_rest])
                r = 0.5 * (r_rest + _cov(zf[other])) if other.sum() >= 8 else r_rest
                stats[(dev, f, h)] = (_cov(zf[sig]), r)
    return PatternScorer(stats, feats.positions, n_targets, feats.harmonics)


# -- offline analysis of a recording ------------------------------------------------------------------
@dataclass
class Analysis:
    window_s: float
    hop_s: float
    harmonics: int
    devices: list[str]
    positions: dict[str, list[str]]      # device -> positions of its decoding columns (+ noise refs)
    t_end: np.ndarray                    # [N]
    used: np.ndarray                     # [N, D] device decoded in this window
    rho: np.ndarray                      # [N, D, F]
    z: dict[str, np.ndarray]             # device -> [N, F, H, C]


def analyse(t: np.ndarray, x: np.ndarray, present: np.ndarray, layout: Layout, fs: float,
            targets: Mapping[str, float], window_s: float, hop_s: float = config.SSVEP_HOP_S,
            harmonics: int = config.SSVEP_HARMONICS, filtered: Optional[tuple[np.ndarray, np.ndarray]] = None
            ) -> Analysis:
    """Windows over a raw recording (µV, layout.positions order), decoded exactly like live."""
    if filtered is None:
        filtered = filter_recording(x, fs)
    xs, xd = filtered
    check = xs if config.SSVEP_ARTIFACT_IN_DECODED_BAND else xd
    muscle = MuscleDetector(fs, x.shape[1]).process(x)
    present = np.asarray(present, dtype=float).reshape(len(t), -1)
    if present.shape[1] != len(layout.device_names):
        present = np.ones((len(t), len(layout.device_names)))
    fe = FeatureExtractor(fs, targets, harmonics)
    devs = list(layout.device_names)
    full = {d: layout.columns(k) for k, d in enumerate(devs)}
    positions = {d: [layout.positions[i] for i in full[d]] for d in devs}
    ends = np.arange(t[0] + window_s, t[-1] + 1e-9, hop_s)
    i1 = np.searchsorted(t, ends + 1e-6, side="right")
    i0 = np.searchsorted(t, ends - window_s + 1e-6, side="right")
    n_min = int(window_s * fs * 0.75)
    F = len(targets)
    N = len(ends)
    used = np.zeros((N, len(devs)), dtype=bool)
    rho = np.full((N, len(devs), F), np.nan)
    z = {d: np.zeros((N, F, harmonics, len(full[d])), dtype=complex) for d in devs}
    for w in range(N):
        a, b = i0[w], i1[w]
        if b - a < n_min:
            continue
        groups, _, masks = layout.groups(check[a:b], present[a:b].mean(axis=0), muscle=muscle[a:b],
                                         max_blanked=config.SSVEP_MAX_BLANKED, fs=fs)
        feats = fe.window(xs[a:b], groups, masks=masks)
        for k, d in enumerate(devs):
            if d in groups:
                used[w, k] = True
                rho[w, k], z[d][w] = feats[d]
    return Analysis(window_s, hop_s, harmonics, devs, positions, ends, used, rho, z)


def filter_recording(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    return (StreamFilter(fs, config.SSVEP_BAND, config.EEG_NOTCH_HZ).process(x),
            StreamFilter(fs, config.EEG_DISPLAY_BAND, config.EEG_NOTCH_HZ).process(x))


def device_key(devs: Sequence[str]) -> str:
    return "+".join(devs)


def cca_norm_stats(an: Analysis, rest_mask: np.ndarray) -> dict[str, list[float]]:
    """Mean/sd of the (device-averaged) CCA score in centre-gaze windows, per set of devices, and
    for each device alone, so one threshold works whichever devices are decoding."""
    out = {}
    keys = {}
    for w in np.flatnonzero(rest_mask & an.used.any(axis=1)):
        keys.setdefault(tuple(np.flatnonzero(an.used[w])), []).append(w)
    for ks, ws in keys.items():
        vals = np.nanmean(an.rho[ws][:, list(ks), :], axis=1).ravel()
        if len(vals) >= 20:
            out[device_key([an.devices[k] for k in ks])] = [float(vals.mean()), float(vals.std() + 1e-6)]
    for k, d in enumerate(an.devices):
        ws = np.flatnonzero(rest_mask & an.used[:, k])
        if len(ws) >= 5:
            vals = an.rho[ws, k, :].ravel()
            out[d] = [float(vals.mean()), float(vals.std() + 1e-6)]
    return out


def cca_scores(an: Analysis, norm: Mapping[str, Sequence[float]], rows: Optional[np.ndarray] = None) -> np.ndarray:
    rows = np.arange(len(an.t_end)) if rows is None else rows
    out = np.full((len(rows), an.rho.shape[2]), np.nan)
    fallback = norm.get(device_key(an.devices)) or next(iter(norm.values()), [0.0, 1.0])
    for j, w in enumerate(rows):
        ks = np.flatnonzero(an.used[w])
        if len(ks) == 0:
            continue
        mu, sd = norm.get(device_key([an.devices[k] for k in ks]), fallback)
        out[j] = (np.mean(an.rho[w, ks, :], axis=0) - mu) / sd
    return out


def pattern_scores(an: Analysis, scorer: PatternScorer, rows: np.ndarray) -> np.ndarray:
    out = np.full((len(rows), an.rho.shape[2]), np.nan)
    for j, w in enumerate(rows):
        ks = np.flatnonzero(an.used[w])
        if len(ks) == 0:
            continue
        out[j] = sum(scorer.llr(an.devices[k], an.positions[an.devices[k]], an.z[an.devices[k]][w]) for k in ks)
    return out


# -- replaying the smoother -------------------------------------------------------------------------
def simulate(t_end: np.ndarray, lab: np.ndarray, n_smooth: int, rest: int,
             deadman: float = config.DEADMAN_S) -> np.ndarray:
    """What pipeline/smoother.py outputs at each window end. ``lab``: decided class per window
    (target index, ``rest``, or -1 = no prediction). Returns the command class per window."""
    N = len(t_end)
    out = np.full(N, rest)
    valid = np.flatnonzero(lab >= 0)
    if len(valid) == 0:
        return out
    tv, lv = t_end[valid], lab[valid]
    gap = np.diff(tv, prepend=-np.inf) > deadman + 1e-9          # dead-man: start over at S
    new = gap.copy()
    new[1:] |= lv[1:] != lv[:-1]
    starts = np.flatnonzero(new)
    run = np.arange(len(lv)) - starts[np.cumsum(new) - 1] + 1
    adopt = np.where(run >= n_smooth, lv, -1)
    adopt = np.where(gap & (adopt < 0), rest, adopt)
    last = np.maximum.accumulate(np.where(adopt >= 0, np.arange(len(lv)), -1))
    cmd_v = np.where(last >= 0, adopt[np.maximum(last, 0)], rest)
    pos = np.searchsorted(valid, np.arange(N), side="right") - 1
    has = pos >= 0
    p = np.maximum(pos, 0)
    cmd = np.where(has, cmd_v[p], rest)
    stale = ~has | (t_end - tv[p] > deadman + 1e-9)
    return np.where(stale, rest, cmd)


@dataclass
class Timeline:
    """The held-out windows in time order, with what the user was asked to do at each."""
    t_end: np.ndarray
    trial: np.ndarray          # trial index per window
    want: np.ndarray           # class index wanted (rest = F)
    prev: np.ndarray           # class index asked for in the previous trial
    starts: np.ndarray         # first window of each trial (in this timeline)
    onsets: np.ndarray         # per trial in timeline
    trial_s: np.ndarray
    is_rest: np.ndarray        # per trial
    tpos: np.ndarray           # per window: its trial's position in this timeline


def make_timeline(t_end: np.ndarray, rows: np.ndarray, trials: Sequence[Trial], tri_of: np.ndarray,
                  labels: Sequence[str]) -> Timeline:
    idx = {l: i for i, l in enumerate(labels)}
    rest = len(labels)
    cls = lambda l: idx.get(l, rest)  # noqa: E731
    tr = tri_of[rows]
    _, starts = np.unique(tr, return_index=True)
    starts = np.sort(starts)
    ids = tr[starts]
    return Timeline(t_end=t_end[rows], trial=tr, want=np.array([cls(trials[i].label) for i in tr]),
                    prev=np.array([cls(trials[i].prev) for i in tr]), starts=starts,
                    onsets=np.array([trials[i].onset for i in ids]),
                    trial_s=np.array([trials[i].end - trials[i].onset for i in ids]),
                    is_rest=np.array([trials[i].label == REST for i in ids]),
                    tpos=np.searchsorted(starts, np.arange(len(rows)), side="right") - 1)


def score_run(tl: Timeline, cmd: np.ndarray, rest: int) -> dict:
    """Per-trial time to the right command, misses, and the share of time with a wrong drive command."""
    t = tl.t_end
    hit_t = np.minimum.reduceat(np.where(cmd == tl.want, t, np.inf), tl.starts)
    t_hit = hit_t - tl.onsets
    before = t < hit_t[tl.tpos]                 # still waiting for the right command
    # Wrong drive command: not what was asked, and not just the previous trial's command lingering.
    wrong = (cmd != rest) & (cmd != tl.want) & ~(before & (cmd == tl.prev))
    miss = ~np.isfinite(t_hit)
    capped = np.where(miss, tl.trial_s + 1.0, t_hit)
    tgt = ~tl.is_rest
    after_go = ~before & (tl.want != rest)
    at_rest = (tl.want == rest) & ~before
    # Objective: seconds per trial the rover isn't doing what was asked (latency + stutter + wrong).
    hop = float(np.median(np.diff(t))) if len(t) > 1 else 0.1
    off = np.add.reduceat((cmd != tl.want).astype(float), tl.starts) * hop
    return dict(objective=float(np.mean(off)), false=float(np.mean(wrong)),
                go_s=float(np.median(capped[tgt])) if tgt.any() else math.nan,
                stop_s=float(np.median(capped[~tgt])) if (~tgt).any() else math.nan,
                hit=float(np.mean(~miss[tgt])) if tgt.any() else math.nan,
                drop=float(np.mean(cmd[after_go] == rest)) if after_go.any() else 0.0,
                rest_drive=float(np.mean(cmd[at_rest] != rest)) if at_rest.any() else 0.0)


def decide(scores: np.ndarray, threshold: float, margin: float, rest: int) -> np.ndarray:
    """Per-window class: best target if it clears the threshold and beats the runner-up by the margin."""
    ok = ~np.isnan(scores).any(axis=1)
    s = np.where(ok[:, None], scores, 0.0)
    order = np.sort(s, axis=1)
    best = order[:, -1]
    second = order[:, -2] if s.shape[1] > 1 else best - np.inf
    lab = np.where((best >= threshold) & (best - second >= margin), np.argmax(s, axis=1), rest)
    return np.where(ok, lab, -1)


def sweep(tl: Timeline, scores: np.ndarray, rest: int, smooth: Sequence[int] = config.SSVEP_CAL_SMOOTH,
          max_false: float = config.SSVEP_CAL_MAX_FALSE,
          max_rest_drive: float = config.SSVEP_CAL_MAX_REST_DRIVE) -> tuple[dict, list[dict]]:
    """Grid over threshold x margin x debounce on one candidate's held-out scores. Returns the best
    setting (least time per trial not doing what was asked, with wrong drive commands <= max_false
    and driving while looking at the centre <= max_rest_drive) and all of them."""
    ok = ~np.isnan(scores).any(axis=1)
    if ok.sum() < 20:
        return {}, []
    s = np.sort(scores[ok], axis=1)
    best, gapv = s[:, -1], s[:, -1] - s[:, -2]
    thresholds = np.unique(np.quantile(best, np.linspace(0.3, 0.995, 36)))
    margins = np.unique(np.quantile(gapv, [0.0, 0.1, 0.2, 0.35, 0.5]))
    margins[0] = 0.0
    rows = []
    for m in margins:
        for th in thresholds:
            lab = decide(scores, th, m, rest)
            for n in smooth:
                r = score_run(tl, simulate(tl.t_end, lab, n, rest), rest)
                r.update(threshold=float(th), margin=float(m), n_smooth=int(n))
                rows.append(r)
    safe = [r for r in rows if r["false"] <= max_false and r["rest_drive"] <= max_rest_drive]
    pool = safe or sorted(rows, key=lambda r: r["false"])[:max(1, len(rows) // 50)]
    # Shortest mean time to the right command; near-ties -> fewer false commands, then more debounce.
    best_obj = min(r["objective"] for r in pool)
    near = [r for r in pool if r["objective"] <= best_obj + 0.05]
    pick = min(near, key=lambda r: (r["false"] + r["rest_drive"], -r["n_smooth"], -r["threshold"]))
    pick = dict(pick, safe=bool(safe))
    return pick, rows


# -- the result -------------------------------------------------------------------------------------
@dataclass
class SsvepCalibration:
    method: str
    window_s: float
    hop_s: float
    n_smooth: int
    threshold: float
    margin: float
    harmonics: int
    targets: dict
    devices: dict                      # device -> positions the model uses (decoding + noise refs)
    cca_norm: dict = field(default_factory=dict)
    pattern: dict = field(default_factory=dict)  # "device|target|harmonic" -> {"S": [re, im], "R": [re, im]}
    report: dict = field(default_factory=dict)
    user: str = "default"
    setup: str = ""
    saved_at: str = ""

    # Interface shared with the older Calibration / AlphaCalibration
    @property
    def rest_recall(self) -> Optional[float]:
        """Share of settled centre-gaze time the rover stays stopped (held-out blocks)."""
        rd = self.report.get("chosen", {}).get("rest_drive")
        return None if rd is None else 1.0 - rd

    def describe(self) -> str:
        c = self.report.get("chosen", {})
        if not c:
            return f"SSVEP {self.method}, {self.window_s:g} s windows"
        warn = "" if c.get("safe", True) else " (WARNING: couldn't reach the false-command target)"
        return (f"{self.method.upper()} decoder, {self.window_s:g} s windows: right command in "
                f"{c['go_s']:.1f} s (stop {c['stop_s']:.1f} s), {c['hit'] * 100:.0f}% of trials, "
                f"wrong drive {c['false'] * 100:.1f}% of the time{warn}")

    def scorer(self) -> "LiveScorer":
        return LiveScorer(self)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=1)

    @classmethod
    def from_dict(cls, d: dict) -> "SsvepCalibration":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _pack(m: np.ndarray) -> list:
    return [np.real(m).round(10).tolist(), np.imag(m).round(10).tolist()]


def _unpack(v) -> np.ndarray:
    return np.array(v[0]) + 1j * np.array(v[1])


class LiveScorer:
    """Scores for one live window from FeatureExtractor output: (scores[F] or None)."""

    def __init__(self, cal: SsvepCalibration) -> None:
        self.cal = cal
        self.labels = list(cal.targets)
        self.needs_cca = cal.method == "cca"
        self.pattern = None
        if cal.method == "pattern":
            stats = {}
            for key, v in cal.pattern.items():
                dev, f, h = key.split("|")
                stats[(dev, int(f), int(h))] = (_unpack(v["S"]), _unpack(v["R"]))
            self.pattern = PatternScorer(stats, cal.devices, len(self.labels), cal.harmonics)

    def scores(self, feats: Mapping[str, tuple], positions: Mapping[str, Sequence[str]],
               device_order: Sequence[str]) -> Optional[np.ndarray]:
        devs = [d for d in device_order if d in feats]
        if not devs:
            return None
        if self.pattern is not None:
            return sum(self.pattern.llr(d, positions[d], feats[d][1]) for d in devs)
        norm = self.cal.cca_norm
        fallback = norm.get(device_key(device_order)) or next(iter(norm.values()), [0.0, 1.0])
        mu, sd = norm.get(device_key(devs), fallback)
        return (np.mean([feats[d][0] for d in devs], axis=0) - mu) / sd


# -- fitting ---------------------------------------------------------------------------------------
def fit(t: np.ndarray, x: np.ndarray, present: Optional[np.ndarray], segments: Sequence[tuple[str, float, float]],
        layout: Layout, fs: float, targets: Mapping[str, float], *, methods: Sequence[str] = config.SSVEP_CAL_METHODS,
        windows: Sequence[float] = config.SSVEP_CAL_WINDOWS, hop_s: float = config.SSVEP_HOP_S,
        harmonics: int = config.SSVEP_HARMONICS, max_false: float = config.SSVEP_CAL_MAX_FALSE,
        max_rest_drive: float = config.SSVEP_CAL_MAX_REST_DRIVE, progress=None) -> SsvepCalibration:
    """Fit and choose the decoder from one calibration recording (see the module docstring)."""
    labels = list(targets)
    rest = len(labels)
    trials = trials_from_segments(segments)
    if not trials or REST not in {tr.label for tr in trials} or len({tr.label for tr in trials}) < 2:
        raise ValueError("calibration needs centre (rest) trials and at least one box")
    blocks = sorted({tr.block for tr in trials})
    if len(blocks) < 2:
        raise ValueError("calibration needs at least 2 blocks to check itself")
    present = np.ones((len(t), len(layout.device_names))) if present is None else present
    filtered = filter_recording(x, fs)
    results, analyses = [], {}
    for wi, window_s in enumerate(windows):
        if progress:
            progress(f"analysing {window_s:g} s windows", wi / len(windows))
        an = analyse(t, x, present, layout, fs, targets, window_s, hop_s, harmonics, filtered)
        tri_of, settled = _assign(an.t_end, trials, window_s)
        label_idx = np.array([(labels.index(trials[i].label) if trials[i].label in labels else rest)
                              if i >= 0 else -1 for i in tri_of])
        in_trial = tri_of >= 0
        rows = np.flatnonzero(in_trial)
        if len(rows) < 50 or not any(an.used[settled & (label_idx == rest)].any(axis=1)):
            continue
        block_of = np.array([trials[i].block if i >= 0 else -1 for i in tri_of])
        analyses[window_s] = (an, label_idx, settled, block_of, rows, tri_of)
        tl = make_timeline(an.t_end, rows, trials, tri_of, labels)
        for method in methods:
            sc = np.full((len(rows), rest), np.nan)
            for b in blocks:                              # leave one block out
                train = settled & (block_of != b) & (label_idx >= 0)
                test = block_of[rows] == b
                if method == "pattern":
                    model = fit_pattern(an, train, label_idx, rest)
                    if not model.stats:
                        continue
                    sc[test] = pattern_scores(an, model, rows[test])
                else:
                    norm = cca_norm_stats(an, train & (label_idx == rest))
                    if not norm:
                        continue
                    sc[test] = cca_scores(an, norm, rows[test])
            pick, _ = sweep(tl, sc, rest, max_false=max_false, max_rest_drive=max_rest_drive)
            if pick:
                results.append(dict(pick, method=method, window_s=window_s))
    if progress:
        progress("choosing", 1.0)
    if not results:
        raise ValueError("not enough clean calibration data (artifacts or bad electrodes?)")
    safe = [r for r in results if r["safe"]]
    pool = safe or sorted(results, key=lambda r: r["false"])[:1]
    chosen = min(pool, key=lambda r: (r["objective"], r["false"] + r["rest_drive"]))
    if not chosen["hit"] >= config.SSVEP_CAL_MIN_HIT:
        # Don't install a decoder that never (or rarely) drives: say why instead.
        resp = response_check(t, x, segments, layout, fs, targets, filtered[0])
        best = max((r["hit"] for r in results), default=0.0)
        worst = ", ".join(f"{l} {v:.1f}x" for l, v in resp.items())
        raise ValueError(f"no reliable flicker response: the best setting picked up {best * 100:.0f}% of the boxes "
                         f"without false commands. Looking at a box raised its frequency by {worst} "
                         f"(a response is usually 2x+). Check the boxes are big and steady on screen and "
                         f"that the driver stares at each one; kept the previous decoder")
    an, label_idx, settled, *_ = analyses[chosen["window_s"]]
    train = settled & (label_idx >= 0)
    rest_rows = train & (label_idx == rest)
    cal = SsvepCalibration(method=chosen["method"], window_s=chosen["window_s"], hop_s=hop_s,
                           n_smooth=chosen["n_smooth"], threshold=round(chosen["threshold"], 4),
                           margin=round(chosen["margin"], 4), harmonics=harmonics, targets=dict(targets),
                           devices=an.positions, cca_norm=cca_norm_stats(an, rest_rows))
    if cal.method == "pattern":
        model = fit_pattern(an, train, label_idx, rest)
        cal.pattern = {f"{d}|{f}|{h}": {"S": _pack(S), "R": _pack(R)} for (d, f, h), (S, R) in model.stats.items()}
    keep = ("method", "window_s", "objective", "go_s", "stop_s", "hit", "false", "drop", "rest_drive",
            "threshold", "margin", "n_smooth", "safe")
    cal.report = {"chosen": {k: chosen[k] for k in keep},
                  "candidates": [{k: r[k] for k in keep} for r in sorted(results, key=lambda r: r["objective"])],
                  "trials": len(trials), "blocks": len(blocks)}
    return cal


def _assign(t_end: np.ndarray, trials: Sequence[Trial], window_s: float,
            settle_s: float = config.SSVEP_CAL_SETTLE_S) -> tuple[np.ndarray, np.ndarray]:
    """Trial index of each window (by its end time; -1 outside trials), and whether the whole window
    lies inside its trial after the settle time (usable for training)."""
    onsets = np.array([tr.onset for tr in trials])
    ends = np.array([tr.end for tr in trials])
    i = np.searchsorted(onsets, t_end, side="right") - 1
    inside = (i >= 0) & (t_end <= ends[np.maximum(i, 0)])
    tri = np.where(inside, i, -1)
    settled = inside & (t_end - window_s >= onsets[np.maximum(i, 0)] + settle_s)
    return tri, settled


def save(cal: SsvepCalibration, models_dir: Path) -> Path:
    cal.saved_at = time.strftime("%Y-%m-%d %H:%M:%S")
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in cal.user) or "default"
    path = Path(models_dir) / f"ssvep_{safe}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cal.to_json())
    return path


# -- is there a flicker response at all? ----------------------------------------------------------
def snr_at(psd: np.ndarray, freqs: np.ndarray, hz: float, others: Sequence[float] = ()) -> float:
    """Power at ``hz`` over the mean power of neighbouring frequencies (other targets excluded)."""
    lo, hi = config.SSVEP_SNR_NEIGHBOURS_HZ
    k = int(np.argmin(np.abs(freqs - hz)))
    d = np.abs(freqs - hz)
    nb = (d > lo) & (d < hi)
    for o in others:
        nb &= np.abs(freqs - o) > lo
    return float(psd[k] / (np.mean(psd[nb]) + 1e-12)) if nb.any() else 1.0


def response_check(t, x, segments, layout: Layout, fs: float, targets: Mapping[str, float],
                   xs: Optional[np.ndarray] = None) -> dict[str, float]:
    """For each box: how many times its frequency (and double) rose at the decoding electrodes while
    the user looked at it, compared with looking at the centre. ~1x = no response."""
    from scipy import signal as sps

    xs = filter_recording(x, fs)[0] if xs is None else xs
    dec = list(layout.decode_idx)
    trials = trials_from_segments(segments)

    def psd(label):
        parts = [xs[a + int(fs):b] for tr in trials if tr.label == label
                 for a, b in [np.searchsorted(t, [tr.onset, tr.end])] if b - a > 2 * fs]
        if not parts:
            return None, None
        f, P = sps.welch(np.vstack(parts)[:, dec], fs=fs, nperseg=int(2 * fs), axis=0)
        return f, P.mean(axis=1)

    f, rest = psd(REST)
    out = {}
    for label, hz in targets.items():
        f2, P = psd(label)
        if f is None or f2 is None:
            continue
        ratio = [P[np.argmin(np.abs(f - h * hz))] / rest[np.argmin(np.abs(f - h * hz))] for h in (1, 2)]
        out[label] = float(np.mean(ratio))
    return out
