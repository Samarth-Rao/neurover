"""SSVEP decoding with CCA, plus per-user calibration of the decision threshold and margin.

Each target flickers at its own frequency. For a window of (band-passed) EEG
X[n, channels] we compute, for every target, the largest canonical correlation
between X and sine/cosine references at the target frequency and its harmonics.
The best target wins if its correlation clears ``threshold`` and beats the
runner-up by ``margin``; otherwise the output is "rest" (-> S).

With several EEG devices (e.g. a BrainBit Flex + the BrainBit headband) the electrodes come in
groups, one per device. "fused" scoring runs CCA on each device's electrodes and averages the
correlations; "joint" runs one CCA over all of them. A device can drop out live, so calibration
also fits a threshold + margin for each device on its own, and the decoder uses the one that
matches the devices present in each window.

That is the uncalibrated decoder. Pressing C runs the calibration in pipeline/ssvep_cal.py, which
learns a decoder for this user ("pattern", or z-scored CCA) and picks the window length,
threshold, margin and debounce that are fastest without false commands; ``apply`` installs it.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

import config

REST = "rest"
_ARROWS = {"up": "↑", "down": "↓", "left": "←", "right": "→"}


def target_frequencies(frames: Mapping[str, int], refresh_hz: float) -> dict[str, float]:
    return {label: refresh_hz / n for label, n in frames.items()}


def reference_signals(freq: float, n: int, fs: float, harmonics: int = 2) -> np.ndarray:
    t = np.arange(n) / fs
    cols = []
    for h in range(1, harmonics + 1):
        cols += [np.sin(2 * np.pi * h * freq * t), np.cos(2 * np.pi * h * freq * t)]
    return np.column_stack(cols)


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


@dataclass
class Calibration:
    threshold: float
    margin: float
    balanced_accuracy: Optional[float] = None
    rest_recall: Optional[float] = None
    window_s: float = config.SSVEP_WINDOW_S
    user: str = "default"
    saved_at: str = ""
    setup: str = ""        # electrodes + combining it was made with; a different setup needs a new calibration
    devices: str = ""      # devices present for threshold/margin above ("flex+headband")
    by_devices: dict = field(default_factory=dict)   # devices present -> [threshold, margin, rest_recall]

    def describe(self) -> str:
        rest = f", centre kept as rest {self.rest_recall * 100:.0f}%" if self.rest_recall is not None else ""
        alone = [k for k in self.by_devices if k != self.devices]
        extra = f" (also for {', '.join(alone)} alone)" if alone else ""
        return f"SSVEP threshold {self.threshold:.2f}, margin {self.margin:.2f}{rest}{extra}"


class SsvepDecoder:
    mode = "ssvep"
    needs_stimulus = True
    cal_skip_s = config.SSVEP_CAL_SKIP_S

    def __init__(self, fs: float, targets: Mapping[str, float], *, harmonics: int = config.SSVEP_HARMONICS,
                 threshold: Optional[float] = None, margin: Optional[float] = None,
                 groups: Sequence[str] = (), combine: str = config.SSVEP_COMBINE, setup: str = "",
                 positions: Sequence[str] = ()) -> None:
        if combine not in ("fused", "joint"):
            raise ValueError(f"SSVEP_COMBINE must be 'fused' or 'joint', not {combine!r}")
        fused = len(groups) > 1 and combine == "fused"
        if threshold is None:
            threshold = config.SSVEP_THRESHOLD_FUSED if fused else config.SSVEP_THRESHOLD
        if margin is None:
            margin = config.SSVEP_MARGIN_FUSED if fused else config.SSVEP_MARGIN
        self.fs = fs
        self.targets = dict(targets)
        self.labels = tuple(self.targets)
        self.classes = self.labels
        self.command_map = dict(config.SSVEP_COMMANDS)
        self.class_names = {l: f"{_ARROWS.get(l, '')} {f:.1f} Hz".strip() for l, f in self.targets.items()}
        self.class_names[REST] = "Rest"
        self.harmonics = harmonics
        self.threshold = threshold
        self.margin = margin
        self.groups = tuple(groups)       # device names, in order
        self.combine = combine
        self.setup = setup
        self._main = (threshold, margin)  # for the full set of devices (or the only one)
        # Until calibrated, a device decoding on its own uses the single-CCA defaults.
        self._by_devices: dict[str, tuple[float, float]] = (
            {g: (config.SSVEP_THRESHOLD, config.SSVEP_MARGIN) for g in self.groups} if fused else {})
        self.active = self.full_key       # devices used for the latest window
        self.collect_subsets = False      # while calibrating: also score each device alone
        self.last_subsets: dict[str, dict[str, float]] = {}
        self._refs: dict[int, list[np.ndarray]] = {}
        self.positions = tuple(positions)  # column -> electrode position (for the calibrated decoder)
        self.model = None                  # ssvep_cal.SsvepCalibration once calibrated
        self._scorer = None
        self._features = None

    @property
    def score_name(self) -> str:
        if self.model is None:
            return "ρ"
        return "LLR" if self.model.method == "pattern" else "z"

    @property
    def score_scale(self) -> float:
        """Score that fills a dashboard bar (raw CCA correlations are 0-1; calibrated scores aren't)."""
        if self.model is None:
            return 1.0
        return max(2.0 * self.threshold, 1e-6)

    def _references(self, n: int) -> list[np.ndarray]:
        refs = self._refs.get(n)
        if refs is None:
            refs = [reference_signals(f, n, self.fs, self.harmonics) for f in self.targets.values()]
            self._refs = {n: refs}  # window length is ~constant; keep one entry
        return refs

    @property
    def full_key(self) -> str:
        return "+".join(self.groups)

    def key(self, groups: Mapping[str, Sequence[int]]) -> str:
        return "+".join([g for g in self.groups if g in groups] + [g for g in groups if g not in self.groups])

    def _cca(self, window: np.ndarray) -> dict[str, float]:
        refs = self._references(len(window))
        return {label: cca_max_corr(window, ref) for label, ref in zip(self.labels, refs)}

    def _rho(self, window: np.ndarray, groups: Mapping[str, Sequence[int]],
             masks: Optional[Mapping[str, np.ndarray]] = None) -> dict[str, dict[str, float]]:
        """CCA correlations per device (blanked samples left out, slow drift fitted out)."""
        from pipeline import ssvep_cal

        if getattr(self, "_fe", None) is None:
            self._fe = ssvep_cal.FeatureExtractor(self.fs, self.targets, self.harmonics)
        feats = self._fe.window(window, groups, need_z=False, masks=masks)
        return {g: dict(zip(self.labels, (float(v) for v in feats[g][0]))) for g in groups}

    def scores(self, window: np.ndarray, groups: Optional[Mapping[str, Sequence[int]]] = None,
               masks: Optional[Mapping[str, np.ndarray]] = None) -> dict[str, float]:
        """Scores for one window. ``groups``: device -> its columns in ``window`` (None = one group);
        ``masks``: device -> samples to keep."""
        if not groups:
            groups = {"": list(range(window.shape[1]))}
        if self.combine == "joint" or len(groups) == 1:
            cols = sorted({c for g in groups.values() for c in g})
            keep = [m for g, m in (masks or {}).items() if g in groups]
            mask = np.logical_and.reduce(keep) if keep else None
            return self._rho(window, {"_": cols}, None if mask is None else {"_": mask})["_"]
        per = self._rho(window, groups, masks)
        return {label: float(np.mean([p[label] for p in per.values()])) for label in self.labels}

    def _subset_scores(self, window: np.ndarray, groups: Mapping[str, Sequence[int]],
                       masks: Optional[Mapping[str, np.ndarray]] = None) -> dict[str, dict[str, float]]:
        """Scores for the devices present together, and for each of them alone (for calibration)."""
        single = self._rho(window, groups, masks)
        out = {self.key({g: 0}): s for g, s in single.items()}
        if len(groups) > 1:
            if self.combine == "joint":
                out[self.key(groups)] = self.scores(window, groups, masks)
            else:
                out[self.key(groups)] = {l: float(np.mean([s[l] for s in single.values()])) for l in self.labels}
        return out

    def thresholds_for(self, key: str) -> tuple[float, float]:
        return self._by_devices.get(key, self._main)

    def decide(self, scores: Mapping[str, float]) -> tuple[str, float]:
        """(label, confidence). Label is a target or REST; confidence is the best correlation."""
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best, rho = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        if rho < self.threshold or rho - second < self.margin:
            return REST, rho
        return best, rho

    def apply(self, cal) -> None:
        """Install a calibration: an ssvep_cal.SsvepCalibration (learned decoder) or a Calibration
        (thresholds for the uncalibrated CCA scores)."""
        from pipeline import ssvep_cal

        self.threshold, self.margin = cal.threshold, cal.margin
        self._main = (cal.threshold, cal.margin)
        if isinstance(cal, ssvep_cal.SsvepCalibration):
            self.model, self._scorer = cal, cal.scorer()
            self._features = ssvep_cal.FeatureExtractor(self.fs, self.targets, cal.harmonics)
            self._by_devices = {}
            return
        self.model = self._scorer = None
        self._by_devices = {k: (float(v[0]), float(v[1])) for k, v in (cal.by_devices or {}).items()}

    # -- shared EEG decoder interface (see sources/eeg_common.py) ------------------------
    def evaluate(self, xs: np.ndarray, xd: np.ndarray, occipital: Sequence[int],
                 groups: Optional[Mapping[str, Sequence[int]]] = None,
                 masks: Optional[Mapping[str, np.ndarray]] = None) -> tuple[str, float, dict]:
        """``groups``: device -> columns of ``xs`` to decode from (devices missing from this window left
        out); ``masks``: device -> samples to keep (artifacts blanked)."""
        if not groups:
            groups = {self.groups[0] if self.groups else "": list(range(xs.shape[1]))}
        self.active = self.key(groups)
        if self.model is not None:
            return self._evaluate_calibrated(xs, groups, masks)
        if self.collect_subsets and len(self.groups) > 1:
            self.last_subsets = self._subset_scores(xs, groups, masks)
            scores = self.last_subsets[self.active]
        else:
            scores = self.scores(xs, groups, masks)
        self.threshold, self.margin = self.thresholds_for(self.active)
        label, conf = self.decide(scores)
        return label, conf, scores

    def _evaluate_calibrated(self, xs: np.ndarray, groups: Mapping[str, Sequence[int]],
                             masks: Optional[Mapping[str, np.ndarray]] = None) -> tuple[str, float, dict]:
        sc = self._scorer
        feats = self._features.window(xs, groups, need_cca=sc.needs_cca, need_z=sc.pattern is not None,
                                      masks=masks)
        pos = {g: [self.positions[i] if i < len(self.positions) else str(i) for i in cols] for g, cols in groups.items()}
        vals = sc.scores(feats, pos, self.groups or tuple(groups))
        scores = {l: float(v) for l, v in zip(self.labels, vals)} if vals is not None else {l: 0.0 for l in self.labels}
        self.threshold, self.margin = self.model.threshold, self.model.margin
        label, conf = self.decide(scores)
        return label, conf, scores

    def cal_entry(self, scores: Mapping[str, float]):
        """What to keep from one calibration window: with several devices, every device combination."""
        return dict(self.last_subsets) if len(self.groups) > 1 else dict(scores)

    def calibration_segments(self) -> list[tuple[str, float, str]]:
        from pipeline import ssvep_cal

        return ssvep_cal.protocol(self.targets)

    def calibrate(self, samples: Mapping[str, Sequence[Mapping]]) -> Calibration:
        """Thresholds for the uncalibrated CCA scores from labelled score samples (older path)."""
        cal = self.fit_samples(samples)
        self.apply(cal)
        return cal

    def fit_samples(self, samples: Mapping[str, Sequence[Mapping]]) -> Calibration:
        samples = {c: v for c, v in samples.items() if c == REST or c in self.targets}
        entries = [e for v in samples.values() for e in v]
        if entries and isinstance(next(iter(entries[0].values()), None), Mapping):
            cal = self._calibrate_by_devices(samples)
        else:
            cal = calibrate(samples, self.labels)
            cal.devices = self.full_key
        cal.setup = self.setup
        return cal

    def _calibrate_by_devices(self, samples: Mapping[str, Sequence[Mapping[str, Mapping[str, float]]]]) -> Calibration:
        keys = sorted({k for v in samples.values() for e in v for k in e}, key=lambda k: -k.count("+"))
        found: dict[str, Calibration] = {}
        for key in keys:
            sub = {c: [e[key] for e in v if key in e] for c, v in samples.items()}
            try:
                found[key] = calibrate(sub, self.labels)
            except ValueError:
                continue
        if not found:
            raise ValueError("calibration needs centre (rest) windows and at least one target")
        main = self.full_key if self.full_key in found else max(found, key=lambda k: k.count("+"))
        cal = found[main]
        cal.devices = main
        cal.by_devices = {k: [c.threshold, c.margin, c.rest_recall] for k, c in found.items()}
        return cal

    def save(self, cal, models_dir: Path = config.MODELS_DIR) -> Path:
        from pipeline import ssvep_cal

        if isinstance(cal, ssvep_cal.SsvepCalibration):
            return ssvep_cal.save(cal, models_dir)
        return save_calibration(cal, models_dir)

    def load(self, user: str, window_s: Optional[float], models_dir: Path = config.MODELS_DIR):
        cal, msg = load_calibration(user, window_s, models_dir, self.setup)
        if cal is not None:
            self.apply(cal)
        return cal, msg


def calibrate(samples: Mapping[str, Sequence[Mapping[str, float]]], labels: Sequence[str],
              min_rest_recall: float = config.SSVEP_CAL_MIN_REST_RECALL) -> Calibration:
    """Choose threshold + margin from labelled windows.

    ``samples`` maps a condition (REST or a target label) to the score dicts of the
    windows recorded while the user looked there. Safety first: among settings that
    keep at least ``min_rest_recall`` of the centre-gaze windows as rest (the rover
    must not drive on its own), pick the one detecting the targets best. If no setting
    achieves that, pick the one with the highest rest recall.
    """
    labels = list(labels)
    conds = [c for c in [REST, *labels] if samples.get(c)]
    if REST not in conds or len(conds) < 2:
        raise ValueError("calibration needs centre (rest) windows and at least one target")
    S = np.array([[s[l] for l in labels] for c in conds for s in samples[c]])
    truth = np.array([(-1 if c == REST else labels.index(c)) for c in conds for _ in samples[c]])
    order = np.sort(S, axis=1)
    best_rho = order[:, -1]
    gap = order[:, -1] - order[:, -2] if S.shape[1] > 1 else best_rho
    best_idx = np.argmax(S, axis=1)

    thresholds = np.arange(0.05, 0.95, 0.005)
    margins = np.arange(0.0, 0.30, 0.005)
    passes = (best_rho[None, None, :] >= thresholds[:, None, None]) & (gap[None, None, :] >= margins[None, :, None])
    pred = np.where(passes, best_idx[None, None, :], -1)
    correct = pred == truth[None, None, :]
    recalls = np.stack([correct[:, :, truth == (-1 if c == REST else labels.index(c))].mean(axis=2) for c in conds])
    bal = recalls.mean(axis=0)
    rest_recall = recalls[conds.index(REST)]
    target_recall = np.delete(recalls, conds.index(REST), axis=0).mean(axis=0)
    safe = rest_recall >= min_rest_recall
    if not safe.any():
        safe = rest_recall >= rest_recall.max() - 1e-9
    # Best target detection among safe settings; ties -> higher threshold, then higher margin.
    key = np.where(safe, target_recall * 1e4 + thresholds[:, None] * 10 + margins[None, :], -np.inf)
    i, j = np.unravel_index(np.argmax(key), key.shape)
    return Calibration(threshold=round(float(thresholds[i]), 3), margin=round(float(margins[j]), 3),
                       balanced_accuracy=float(bal[i, j]), rest_recall=float(rest_recall[i, j]))


def calibration_path(user: str, models_dir: Path = config.MODELS_DIR) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in user) or "default"
    return Path(models_dir) / f"ssvep_{safe}.json"


def save_calibration(cal: Calibration, models_dir: Path = config.MODELS_DIR) -> Path:
    cal.saved_at = time.strftime("%Y-%m-%d %H:%M:%S")
    path = calibration_path(cal.user, models_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cal.__dict__, indent=2))
    return path


def load_calibration(user: str, window_s: Optional[float], models_dir: Path = config.MODELS_DIR,
                     setup: str = "") -> tuple[Optional[Calibration], Optional[str]]:
    """``window_s`` None = accept the calibration's own window length."""
    from pipeline import ssvep_cal

    path = calibration_path(user, models_dir)
    if not path.exists():
        return None, None
    try:
        data = json.loads(path.read_text())
        cal = ssvep_cal.SsvepCalibration.from_dict(data) if "method" in data else Calibration(**data)
    except Exception as exc:
        return None, f"couldn't read {path.name}: {exc}"
    if window_s is not None and abs(cal.window_s - window_s) > 1e-6:
        return None, f"ignored {path.name}: calibrated for {cal.window_s:g} s windows, now {window_s:g} s"
    if (cal.setup or "") != (setup or ""):
        return None, (f"ignored {path.name}: made with different electrodes "
                      f"({cal.setup or 'an older setup'}); press C to calibrate")
    if isinstance(cal, ssvep_cal.SsvepCalibration):
        return cal, f"loaded SSVEP calibration for '{user}': {cal.describe()}"
    return cal, f"loaded SSVEP calibration for '{user}' (threshold {cal.threshold:.2f}, margin {cal.margin:.2f})"
