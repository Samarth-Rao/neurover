"""SSVEP: which flicker box is the driver looking at?

Score, per device and window: CCA between the device's usable electrodes and sine/cosine references
at each box's frequency and its double, rho[box]. The same CCA at nearby frequencies where nothing
flickers ("controls") is that device's chance level right now. EEG has more power at low
frequencies, so chance correlations fall with frequency; a straight line through the controls'
rho over log-frequency gives the chance level AT each box's frequency, and

    z[box] = (rho[box] - line(box frequency)) / sd(controls around the line)

(Comparing with the plain mean of the controls instead made the lowest box, 8.6 Hz, win half of the
windows at rest, real and mock.) A noisy device then scores ~0 for every box instead of looking
confident, and z means the same for any number of electrodes. Samples are weighted by eeg/clean.py, so spikes count less, and the
references follow each sample's own timestamp (gaps keep the right phase).

Devices pool by Stouffer's sum, z = sum(z_device) / sqrt(devices): devices that agree add up, a
device with nothing to say dilutes rather than vetoes, and a device that drops out just leaves.

Decision: the best box if its z clears ``threshold`` and beats the runner-up by ``margin``; else rest (S).
The controller's smoother then needs ``n_smooth`` windows in a row before the command changes.

Stop gate: while any device's MotionCheck (eeg/clean.py) sees the driver moving or clenching, and for
STOP_HOLD_S after, the decision is a forced stop: S at once, no debounce, whatever the boxes score.

Calibration (C): the scope's cued-trial design, done as when driving the real rover
(calibration_protocol): look at the rover (rest), turn back (3-2-1, an arrow to the next box), look at
that box, then back to the rover; random order. The recording is replayed, stop gate and all, for every window length x threshold x
margin x debounce through the same smoother the rover uses; the fastest setting whose wrong drive
commands stay under SSVEP_CAL_MAX_FALSE, and that keeps still while the driver looks at the rover, wins.
Nothing else is learned, so the result works for whichever electrodes the devices have on the day.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

import config

from . import cues
from .clean import Cleaner, MotionCheck
from .filters import StreamFilter

REST = "rest"
ARROWS = {"up": "↑", "down": "↓", "left": "←", "right": "→"}
_Z_FLOOR = 0.01          # floor for the controls' spread, so a near-constant set can't blow z up


def target_frequencies(frames: Mapping[str, int] = config.SSVEP_FRAMES,
                       refresh_hz: float = config.SSVEP_REFRESH_HZ) -> dict[str, float]:
    return {label: refresh_hz / n for label, n in frames.items()}


def control_frequencies(targets: Sequence[float], lo: Optional[float] = None, hi: Optional[float] = None) -> list[float]:
    """Frequencies around the boxes' (from 1.5 Hz below the lowest to 2 Hz above the highest, at least
    7-16 Hz) but clear (> 0.45 Hz) of every box's half, fundamental and double."""
    lo = min(7.0, min(targets) - 1.5) if lo is None else lo
    hi = max(16.0, max(targets) + 2.0) if hi is None else hi
    bad = [h * f for f in targets for h in (0.5, 1, 2)]
    grid = [round(float(g), 3) for g in np.arange(lo, hi + 1e-9, 0.35)]
    return [g for g in grid if all(abs(g - b) > 0.45 and abs(2 * g - b) > 0.45 for b in bad)]


def weighted_cca(X: np.ndarray, Y: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    """Largest canonical correlation between the columns of X and Y, samples weighted by ``w``."""
    if w is not None:
        s = np.sqrt(w)[:, None]
        X, Y = X * s, Y * s
        sw = np.sum(w)
        X = X - s * (np.sum(X * s, axis=0) / sw)
        Y = Y - s * (np.sum(Y * s, axis=0) / sw)
    else:
        X, Y = X - X.mean(axis=0), Y - Y.mean(axis=0)
    keep = np.std(X, axis=0) > 1e-9
    if not keep.any():
        return 0.0
    qx, _ = np.linalg.qr(X[:, keep])
    qy, _ = np.linalg.qr(Y)
    s = np.linalg.svd(qx.T @ qy, compute_uv=False)
    return float(min(1.0, s[0])) if s.size else 0.0


class Scorer:
    def __init__(self, targets: Mapping[str, float], harmonics: int = config.SSVEP_HARMONICS) -> None:
        self.labels = list(targets)
        self.freqs = np.array([targets[l] for l in self.labels])
        self.controls = np.array(control_frequencies(self.freqs))
        self.harmonics = harmonics
        # least-squares line over log-frequency: rho_controls -> (intercept, slope)
        A = np.column_stack([np.ones(len(self.controls)), np.log(self.controls)])
        self._fit = np.linalg.pinv(A)
        self._line = A
        self._at = np.column_stack([np.ones(len(self.freqs)), np.log(self.freqs)])
        self._dof = max(1, len(self.controls) - 2)

    def _refs(self, t: np.ndarray, freqs: np.ndarray) -> list[np.ndarray]:
        tt = (t - t[0])[:, None]
        out = []
        for f in freqs:
            ph = 2 * np.pi * f * tt * np.arange(1, self.harmonics + 1)[None, :]
            out.append(np.hstack([np.sin(ph), np.cos(ph)]))
        return out

    def device_z(self, t: np.ndarray, X: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
        """z per box for one device's window (``t`` its sample times, ``X`` [n, electrodes])."""
        rho = np.array([weighted_cca(X, r, w) for r in self._refs(t, self.freqs)])
        ctl = np.array([weighted_cca(X, r, w) for r in self._refs(t, self.controls)])
        coef = self._fit @ ctl
        resid = ctl - self._line @ coef
        sd = max(math.sqrt(float(resid @ resid) / self._dof), _Z_FLOOR)
        return (rho - self._at @ coef) / sd


def pool(zs: Sequence[np.ndarray]) -> np.ndarray:
    return np.sum(zs, axis=0) / math.sqrt(len(zs))


def decide(z: np.ndarray, threshold: float, margin: float) -> int:
    """Index of the winning box, or -1 for rest."""
    order = np.argsort(z)
    best = z[order[-1]]
    second = z[order[-2]] if len(z) > 1 else -np.inf
    return int(order[-1]) if best >= threshold and best - second >= margin else -1


# -- per-device window loop, the same live and offline ---------------------------------------------
class DeviceDecoder:
    """One device: judges electrodes and scores windows (live: ``step`` each hop; offline: replay)."""

    def __init__(self, channels: int, scorer: Scorer) -> None:
        self.cleaner = Cleaner(channels)
        self.motion = MotionCheck(channels)
        self.scorer = scorer
        self.used: list[int] = []
        self.z: Optional[np.ndarray] = None
        self.disturbed: Optional[str] = None       # MOVING / MUSCLE in the newest samples, or None

    def step(self, t: np.ndarray, x: np.ndarray, raw: Optional[np.ndarray], fs: float,
             window_s: float, muscle: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Score one window (``x`` decode-band EEG, ``raw`` as recorded, ``muscle`` the muscle band, all
        the same samples); None if not enough data. Also checks the newest samples for movement."""
        self.z, self.disturbed = None, None
        if len(t) < 0.75 * window_s * fs:
            return None
        self.cleaner.update(float(t[-1]), x, raw)
        self.used = self.cleaner.usable()
        if not self.used:
            return None
        if muscle is not None and len(muscle) == len(x):
            a = int(np.searchsorted(t, t[-1] - config.MOVE_RECENT_S, side="right"))
            self.disturbed = self.motion.update(float(t[-1]), x[a:], muscle[a:], self.used)
        w = self.cleaner.weights(x, self.used)
        self.z = self.scorer.device_z(t, x[:, self.used], w)
        return self.z


class StopGate:
    """Stop while any device is disturbed, and for ``hold_s`` after: the reason (MOVING / MUSCLE) or None."""

    def __init__(self, hold_s: float = config.STOP_HOLD_S) -> None:
        self.hold_s = hold_s
        self.until = -math.inf
        self.why: Optional[str] = None

    def update(self, t: float, reasons: Sequence[Optional[str]]) -> Optional[str]:
        reasons = [r for r in reasons if r]
        if reasons:
            self.until, self.why = t + self.hold_s, reasons[0]
        return self.why if t < self.until else None


# -- settings --------------------------------------------------------------------------------------
@dataclass
class Settings:
    window_s: float = config.SSVEP_WINDOW_S
    threshold: float = config.SSVEP_Z_THRESHOLD
    margin: float = config.SSVEP_Z_MARGIN
    n_smooth: int = config.SSVEP_SMOOTH_WINDOWS
    setup: str = ""            # electrodes it was calibrated with (see setup_key)
    frames: dict = field(default_factory=dict)   # the boxes' flicker (config.SSVEP_FRAMES) it was calibrated with
    user: str = "default"
    saved_at: str = ""
    recording: str = ""        # the calibration recording it came from
    report: dict = field(default_factory=dict)

    @property
    def calibrated(self) -> bool:
        return bool(self.report)

    def describe(self) -> str:
        c = self.report.get("chosen")
        if not c:
            return f"default settings: z ≥ {self.threshold:g}, margin {self.margin:g}, {self.window_s:g} s windows"
        n = self.report.get("calibrations", 1)
        source = f" (from {n} calibrations)" if n > 1 else ""
        return (f"{self.window_s:g} s windows: right command in {c['go_s']:.1f} s (stop {c['stop_s']:.1f} s), "
                f"{c['hit'] * 100:.0f}% of boxes, wrong drive {c['false'] * 100:.1f}% of the time{source}")


def setup_key(devices: Mapping[str, Sequence[str]]) -> str:
    """Which electrodes, where: a calibration made with another setup isn't used."""
    return "; ".join(f"{d}:{','.join(sorted(ps))}" for d, ps in sorted(devices.items()))


def settings_path(user: str, models_dir: Path) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in user) or "default"
    return Path(models_dir) / f"ssvep_{safe}.json"


def save_settings(s: Settings, models_dir: Path = config.MODELS_DIR) -> Path:
    s.saved_at = time.strftime("%Y-%m-%d %H:%M:%S")
    path = settings_path(s.user, models_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(s), indent=1))
    return path


def load_settings(user: str, setup: str, models_dir: Path = config.MODELS_DIR) -> tuple[Optional[Settings], str]:
    path = settings_path(user, models_dir)
    if not path.exists():
        return None, ""
    try:
        data = json.loads(path.read_text())
        s = Settings(**{k: v for k, v in data.items() if k in Settings.__dataclass_fields__})
    except Exception as exc:
        return None, f"couldn't read {path.name} ({exc}); press C to calibrate"
    if "window_s" not in data or "n_smooth" not in data:
        return None, f"{path.name} is from the old decoder; press C to calibrate"
    if s.setup != setup:
        return None, f"ignored {path.name}: made with other electrodes ({s.setup or '?'}); press C to calibrate"
    if s.frames and s.frames != dict(config.SSVEP_FRAMES):
        return None, f"ignored {path.name}: made with other flicker frequencies; press C to calibrate"
    return s, f"loaded SSVEP calibration for '{user}': {s.describe()}"


# -- calibration protocol ---------------------------------------------------------------------------
def calibration_protocol(targets: Optional[Mapping[str, float]] = None) -> cues.Protocol:
    """The scope's cued-trial design, as when driving the real rover: per trial look at the rover (rest,
    low beep), turn back (middle beep; 3-2-1 with an arrow to the next box), then look at that box from
    the high beep until its frame goes. Random order, never 3 of a kind in a row, practice first."""
    labels = tuple(targets or target_frequencies())
    classes = labels if config.SSVEP_CAL_AWAY_S > 0 else (*labels, REST)   # without looking away: cued rest
    return cues.Protocol(task="look", classes=classes, per_class=config.SSVEP_CAL_PER_CLASS,
                         practice=config.SSVEP_CAL_PRACTICE, lead_in_s=config.SSVEP_CAL_LEAD_S,
                         away_s=config.SSVEP_CAL_AWAY_S, away_jitter_s=config.SSVEP_CAL_AWAY_JITTER_S,
                         baseline_s=config.SSVEP_CAL_BASELINE_S, cue_s=config.SSVEP_CAL_CUE_S,
                         pause_s=tuple(config.SSVEP_CAL_PAUSE_S), break_every=config.SSVEP_CAL_BREAK_EVERY,
                         break_s=config.SSVEP_CAL_BREAK_S, tail_s=1.0)


def rest_style(meta: Mapping) -> str:
    """How a calibration rested: "rover" (looked away at it) or "centre" (at the cross). Calibrations of
    the two kinds aren't combined: looking at the centre has the boxes flickering at the edge of vision."""
    return "rover" if float(meta.get("protocol", {}).get("away_s", 0) or 0) > 0 else "centre"


@dataclass
class Trial:
    label: str
    onset: float
    end: float
    prev: str             # what was asked just before (its command may still be lingering)


def calibration_trials(meta: Mapping) -> list[Trial]:
    """What the driver was asked to do, and when. Rest: a cued trial's look at the rover (the 3-2-1
    hand-over after it is left out), or in older calibrations its fixation cross; then its cue as that
    box (or rest). Pauses (blinking) and practice trials are left out.
    Older calibrations stored "segments" (label, start, end) instead."""
    if meta.get("trials"):
        out, prev = [], REST
        for tr in meta["trials"]:
            if tr.get("practice") or "cue" not in tr or "end" not in tr:
                continue
            if "away" in tr:
                out.append(Trial(REST, float(tr["away"]), float(tr.get("baseline", tr["cue"])), prev))
            else:
                out.append(Trial(REST, float(tr["baseline"]), float(tr["cue"]), prev))
            out.append(Trial(str(tr["label"]), float(tr["cue"]), float(tr["end"]), REST))
            prev = str(tr["label"])
        return out
    out, prev = [], REST
    for label, a, b in meta.get("segments", []):
        if label == "break":
            prev = REST
        if label in ("ready", "break"):
            continue
        out.append(Trial(str(label), float(a), float(b), prev))
        prev = str(label)
    return out


# -- offline replay of a recording ----------------------------------------------------------------
def _filtered(t: np.ndarray, x: np.ndarray, fs: float, band: tuple[float, float]) -> np.ndarray:
    """The same causal filter as live, restarted at gaps like a live device's."""
    out = np.zeros_like(x)
    filt = StreamFilter(fs, band, config.EEG_NOTCH_HZ)
    cuts = [0, *(np.flatnonzero(np.diff(t) > 0.5) + 1).tolist(), len(t)]
    for a, b in zip(cuts, cuts[1:]):
        filt.reset()
        out[a:b] = filt.process(x[a:b])
    return out


def replay(rec: dict, window_s: float, hop_s: float = config.SSVEP_HOP_S,
           targets: Optional[Mapping[str, float]] = None,
           t_range: Optional[tuple[float, float]] = None, with_stop: bool = False):
    """Pooled z per window over a recording (eeg/recording.py format), decoded exactly like live:
    (window end times [N], z [N, boxes]; NaN rows where no device had a usable window), and with
    ``with_stop`` also whether the stop gate held each window [N] bool."""
    targets = targets or target_frequencies()
    fs = float(rec["meta"]["fs"])
    scorer = Scorer(targets)
    devs = []
    for d in rec["devices"].values():
        t, x = np.asarray(d["t"], dtype=float), np.asarray(d["x"], dtype=float)
        if len(t) < fs:
            continue
        devs.append((t, x, _filtered(t, x, fs, config.SSVEP_BAND), _filtered(t, x, fs, config.MOVE_MUSCLE_BAND),
                     DeviceDecoder(x.shape[1], scorer)))
    if not devs:
        return (np.zeros(0), np.zeros((0, len(targets))), np.zeros(0, dtype=bool))[:3 if with_stop else 2]
    lo = max(d[0][0] for d in devs) + window_s if t_range is None else t_range[0]
    hi = max(d[0][-1] for d in devs) if t_range is None else t_range[1]
    ends = np.arange(lo, hi + 1e-9, hop_s)
    z = np.full((len(ends), len(targets)), np.nan)
    stop = np.zeros(len(ends), dtype=bool)
    gate = StopGate()
    for i, e in enumerate(ends):
        zs, why = [], []
        for t, x, xf, xm, dd in devs:
            a, b = np.searchsorted(t, [e - window_s, e], side="right")
            if b - a < 2 or e - t[b - 1] > 0.5:
                continue
            zd = dd.step(t[a:b], xf[a:b], x[a:b], fs, window_s, xm[a:b])
            if zd is not None:
                zs.append(zd)
                why.append(dd.disturbed)
        stop[i] = gate.update(float(e), why) is not None
        if zs:
            z[i] = pool(zs)
    return (ends, z, stop) if with_stop else (ends, z)


# -- replaying the smoother -------------------------------------------------------------------------
def simulate(t_end: np.ndarray, lab: np.ndarray, n_smooth: int, rest: int,
             deadman: float = config.DEADMAN_S, forced: Optional[np.ndarray] = None) -> np.ndarray:
    """What the controller's smoother outputs at each window end. ``lab``: decided class per window
    (box index, ``rest``, or -1 = no prediction); ``forced``: windows whose class is adopted at once
    (the stop gate's). Returns the command class per window."""
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
    now = forced[valid] if forced is not None else False
    adopt = np.where((run >= n_smooth) | now, lv, -1)
    adopt = np.where(gap & (adopt < 0), rest, adopt)
    last = np.maximum.accumulate(np.where(adopt >= 0, np.arange(len(lv)), -1))
    cmd_v = np.where(last >= 0, adopt[np.maximum(last, 0)], rest)
    pos = np.searchsorted(valid, np.arange(N), side="right") - 1
    has = pos >= 0
    p = np.maximum(pos, 0)
    cmd = np.where(has, cmd_v[p], rest)
    stale = ~has | (t_end - tv[p] > deadman + 1e-9)
    return np.where(stale, rest, cmd)


def score_run(t: np.ndarray, cmd: np.ndarray, trials: Sequence[Trial], tri: np.ndarray, labels: Sequence[str],
              rest: int) -> dict:
    """Per trial: time to the right command, and how much of the time a wrong drive command was sent."""
    cls = {l: i for i, l in enumerate(labels)}
    want = np.array([cls.get(trials[i].label, rest) for i in tri])
    prev = np.array([cls.get(trials[i].prev, rest) for i in tri])
    starts = np.flatnonzero(np.r_[True, tri[1:] != tri[:-1]])
    ids = tri[starts]
    onsets = np.array([trials[i].onset for i in ids])
    dur = np.array([trials[i].end - trials[i].onset for i in ids])
    is_rest = np.array([trials[i].label == REST for i in ids])
    hit_t = np.minimum.reduceat(np.where(cmd == want, t, np.inf), starts)
    t_hit = hit_t - onsets
    tpos = np.searchsorted(starts, np.arange(len(t)), side="right") - 1
    before = t < hit_t[tpos]
    wrong = (cmd != rest) & (cmd != want) & ~(before & (cmd == prev))   # the previous command lingering is not wrong
    miss = ~np.isfinite(t_hit)
    capped = np.where(miss, dur + 1.0, t_hit)
    hop = float(np.median(np.diff(t))) if len(t) > 1 else 0.1
    off = np.add.reduceat((cmd != want).astype(float), starts) * hop
    at_rest = (want == rest) & ~before
    return dict(objective=float(np.mean(off)), false=float(np.mean(wrong)),
                go_s=float(np.median(capped[~is_rest])) if (~is_rest).any() else math.nan,
                stop_s=float(np.median(capped[is_rest])) if is_rest.any() else math.nan,
                hit=float(np.mean(~miss[~is_rest])) if (~is_rest).any() else math.nan,
                rest_drive=float(np.mean(cmd[at_rest] != rest)) if at_rest.any() else 0.0)


def _timeline(parts: Sequence[tuple[dict, list[Trial]]], window_s: float, hop_s: float,
              targets: Mapping[str, float], progress=None, frac=(0.0, 1.0)):
    """Every recording's windows inside its trials, end to end on one timeline (each shifted after the
    last, with a gap longer than the dead-man between them, so the smoother starts over): (t, z,
    stop gate, trial index per window, the shifted trials)."""
    T, Z, S, TRI, trials_all, offset = [], [], [], [], [], 0.0
    for k, (rec, trials) in enumerate(parts):
        if progress:
            progress(f"replaying {window_s:g} s windows" + (f", calibration {k + 1}/{len(parts)}" if len(parts) > 1 else ""),
                     frac[0] + (frac[1] - frac[0]) * k / len(parts))
        onsets = np.array([tr.onset for tr in trials])
        ends = np.array([tr.end for tr in trials])
        t, z, stop = replay(rec, window_s, hop_s, targets, (onsets[0], ends[-1]), with_stop=True)
        i = np.searchsorted(onsets, t, side="right") - 1
        inside = (i >= 0) & (t <= ends[np.maximum(i, 0)])
        shift = offset - onsets[0]
        T.append(t[inside] + shift)
        Z.append(z[inside])
        S.append(stop[inside])
        TRI.append(i[inside] + len(trials_all))
        trials_all += [Trial(tr.label, tr.onset + shift, tr.end + shift, tr.prev) for tr in trials]
        offset += ends[-1] - onsets[0] + 10.0
    return np.concatenate(T), np.vstack(Z), np.concatenate(S), np.concatenate(TRI), trials_all


def fit(recs, windows: Sequence[float] = config.SSVEP_CAL_WINDOWS, hop_s: float = config.SSVEP_HOP_S,
        smooth: Sequence[int] = config.SSVEP_CAL_SMOOTH, max_false: float = config.SSVEP_CAL_MAX_FALSE,
        max_rest_drive: float = config.SSVEP_CAL_MAX_REST_DRIVE,
        progress: Optional[Callable[[str, float], None]] = None) -> Settings:
    """Choose window length, threshold, margin and debounce from one calibration recording, or several
    with the same electrodes (all their trials together: repeated calibrations add up). See the module
    docstring. ValueError, with the reason, if no setting reaches the boxes safely."""
    recs = [recs] if isinstance(recs, dict) else list(recs)
    targets = target_frequencies()
    labels = list(targets)
    rest = len(labels)

    def same_box(meta, label):     # a box no longer used, or flickering at another frequency then: left out
        frames = meta.get("frames")
        return label in targets and (not frames or frames.get(label) == config.SSVEP_FRAMES.get(label))
    parts = [(rec, tr) for rec in recs for tr in [[x for x in calibration_trials(rec["meta"])
                                                   if x.label == REST or same_box(rec["meta"], x.label)]]
             if any(x.label == REST for x in tr) and len({x.label for x in tr}) >= 2]
    if not parts:
        raise ValueError("the calibration needs centre (rest) trials and at least one box")
    results, n_trials = [], 0
    for wi, window_s in enumerate(windows):
        t, z, stop, tri, trials = _timeline(parts, window_s, hop_s, targets, progress,
                                      (wi / len(windows), (wi + 1) / len(windows)))
        n_trials = len(trials)
        ok = ~np.isnan(z).any(axis=1)
        if ok.sum() < 50:
            continue
        s = np.sort(np.where(ok[:, None], z, 0.0), axis=1)
        best, gap, arg = s[:, -1], s[:, -1] - s[:, -2], np.argmax(np.where(ok[:, None], z, 0.0), axis=1)
        thresholds = np.unique(np.quantile(best[ok], np.linspace(0.3, 0.995, 30)))
        margins = np.unique(np.r_[0.0, np.quantile(gap[ok], [0.1, 0.25, 0.5])])
        for m in margins:
            for th in thresholds:
                lab = np.where((best >= th) & (gap >= m), arg, rest)   # same rule as decide()
                lab[~ok] = -1
                lab[stop] = rest                                        # the stop gate, as live
                for n in smooth:
                    r = score_run(t, simulate(t, lab, n, rest, forced=stop), trials, tri, labels, rest)
                    r.update(window_s=float(window_s), threshold=float(th), margin=float(m), n_smooth=int(n))
                    results.append(r)
    if progress:
        progress("choosing", 1.0)
    if not results:
        raise ValueError("not enough usable calibration data (no device streaming, or every electrode bad?)")
    safe = [r for r in results if r["false"] <= max_false and r["rest_drive"] <= max_rest_drive]
    pool_ = safe or sorted(results, key=lambda r: r["false"] + r["rest_drive"])[:1]
    best_obj = min(r["objective"] for r in pool_)
    near = [r for r in pool_ if r["objective"] <= best_obj + 0.05]
    chosen = min(near, key=lambda r: (r["false"] + r["rest_drive"], -r["n_smooth"], -r["threshold"]))
    chosen["safe"] = bool(safe)
    if not chosen["hit"] >= config.SSVEP_CAL_MIN_HIT:
        best = max(r["hit"] for r in results)
        raise ValueError(f"no reliable flicker response: the best safe setting reached {chosen['hit'] * 100:.0f}% of "
                         f"the boxes (any setting: {best * 100:.0f}%). Check the electrodes sit over the back of "
                         f"the head, the boxes are big and steady, and the driver stares at each one; "
                         f"kept the previous settings")
    keep = ("window_s", "threshold", "margin", "n_smooth", "objective", "go_s", "stop_s", "hit", "false",
            "rest_drive", "safe")
    return Settings(window_s=chosen["window_s"], threshold=round(chosen["threshold"], 3),
                    margin=round(chosen["margin"], 3), n_smooth=chosen["n_smooth"], frames=dict(config.SSVEP_FRAMES),
                    report={"chosen": {k: chosen[k] for k in keep}, "trials": n_trials, "calibrations": len(parts)})
