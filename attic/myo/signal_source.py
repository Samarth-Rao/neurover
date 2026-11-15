"""Base class for sources that stream multi-channel body signals (EMG now, EEG later).

Subclasses only provide timestamped samples via ``_read_samples()``. This class
does the rest of the pipeline and emits one Prediction per window:

    samples -> Windower -> features -> (training capture) -> KNN -> Prediction
"""
from __future__ import annotations

import abc
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

import commands
import config
from pipeline.classifier import FitReport, GestureClassifier, Trainer
from pipeline.features import Windower, extract

from .base import Prediction, Source

_MAX_PENDING = 400


class SignalSource(Source):
    classes = config.GESTURES
    command_map = config.GESTURE_TO_COMMAND
    class_names = config.GESTURE_NAMES
    label_keys = config.GESTURE_KEYS          # training mode: number key -> class
    supports_training = True
    feature_names: Sequence[str] = ("mav",)

    def __init__(self, channels: int, model_name: str, *,
                 window_s: float = config.WINDOW_S, hop_s: float = config.HOP_S,
                 models_dir=config.MODELS_DIR, autoload: bool = True,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.channels = channels
        self.clock = clock
        self.windower = Windower(window_s, hop_s)
        self.trainer = Trainer(model_name, self.classes, models_dir=models_dir, meta={
            "channels": channels, "features": list(self.feature_names),
            "window_s": window_s, "hop_s": hop_s})
        self.recording: Optional[str] = None     # label being captured in training mode
        self._record_since = 0.0
        self.record_settle_s = config.TRAIN_SETTLE_S
        self.recorded_this_hold = 0
        self.min_full_scale = config.EMG_DISPLAY_MIN_FULL_SCALE   # EMG bars never zoom in further
        self.signal_full_scale = self.min_full_scale
        self._peak = 0.0
        # EMG bars: per-channel auto range (recent floor + decaying peak) until a calibration fixes it.
        self._levels: deque = deque(maxlen=max(10, int(config.EMG_DISPLAY_HISTORY_S / hop_s)))
        self._chan_peak = np.zeros(channels)
        self.display_cal: Optional[tuple[np.ndarray, np.ndarray]] = None   # (relaxed level, span) per channel
        self._cal_backup = None
        self.windows = 0
        self._pending: list[Prediction] = []
        self._last: Optional[Prediction] = None
        self.load_message = self.trainer.load() if autoload else None
        disp = (self.trainer.saved_meta or {}).get("display") if autoload else None
        if disp:
            self.display_cal = (np.array(disp["relaxed"], dtype=float), np.array(disp["span"], dtype=float))

    # -- subclass hook -------------------------------------------------------------------
    @abc.abstractmethod
    def _read_samples(self) -> list[tuple[float, Sequence[float]]]:
        """All samples received since the last call, as (time.monotonic() timestamp, values)."""

    # -- Source API ----------------------------------------------------------------------
    def latest(self) -> Optional[Prediction]:
        self._process()
        return self._last

    def poll(self) -> list[Prediction]:
        self._process()
        out, self._pending = self._pending, []
        return out

    # -- training ------------------------------------------------------------------------
    def set_recording(self, label: Optional[str]) -> None:
        """Label incoming windows with ``label`` (None = stop). Called every frame from the UI."""
        if label is not None and label not in self.classes:
            raise ValueError(f"unknown label {label!r}")
        if label != self.recording:
            self.recording = label
            self._record_since = self.clock()
            self.recorded_this_hold = 0

    # -- EMG bars ------------------------------------------------------------------------
    def display_levels(self, level) -> np.ndarray:
        """0-1 per channel for the dashboard bars: from the relaxed level to a strong gesture."""
        level = np.asarray(level, dtype=float)
        if self.display_cal is not None:
            base, span = self.display_cal
        else:
            hist = np.array(self._levels) if self._levels else level[None, :]
            base = np.percentile(hist, 10, axis=0)
            span = self._chan_peak - base
        span = np.maximum(span, 0.25 * self.min_full_scale)
        return np.clip((level - base) / span, 0.0, 1.0)

    # -- guided gesture calibration (C) ---------------------------------------------------
    def calibration_segments(self) -> list[tuple[str, float, str]]:
        """Each class held EMG_CAL_HOLD_S, random order, EMG_CAL_REPS times, relaxed gaps between."""
        names = self.class_names
        segs = [("ready", config.EMG_CAL_LEAD_S,
                 "Pick ONE gesture for each direction (and for stop) and keep using it. Relax for now.")]
        prev = None
        for rep in range(config.EMG_CAL_REPS):
            order = list(self.classes)
            for _ in range(50):
                random.shuffle(order)
                if order[0] != prev:
                    break
            for label in order:
                what = ("RELAX your hand and arm completely" if label == "relax" else
                        f"Do your {names.get(label, label).upper()} gesture and hold it")
                cmd = self.command_map.get(label, "?")
                segs.append((label, config.EMG_CAL_HOLD_S, f"{rep + 1}/{config.EMG_CAL_REPS}: {what}  (→ {cmd})"))
                segs.append(("gap", config.EMG_CAL_GAP_S, "Relax"))
                prev = label
        return segs

    def start_recording(self) -> None:
        """A gesture calibration starts: set the current model aside (restored if it's cancelled)."""
        tr = self.trainer
        self._cal_backup = (tr.data.arrays(), tr.clf.model, tr.report, tr.dirty)
        tr.data.clear()
        self.record_settle_s = config.EMG_CAL_SETTLE_S

    def stop_recording(self):
        self.set_recording(None)
        return None

    def set_calibration_label(self, label: Optional[str]) -> None:
        self.set_recording(label if label in self.classes else None)

    def cancel_calibration(self) -> None:
        self.set_recording(None)
        self.record_settle_s = config.TRAIN_SETTLE_S
        backup, self._cal_backup = self._cal_backup, None
        if backup is not None:
            (X, y), model, report, dirty = backup
            tr = self.trainer
            tr.data.clear()
            if len(y):
                tr.data.extend(X, y)
            tr.clf.model, tr.report, tr.dirty = model, report, dirty

    def fit_calibration(self, recording=None, timeline=None, progress=None) -> "EmgCalibration":
        """Fit a new model on the calibration windows (safe in a worker thread: nothing live changes)."""
        X, y = self.trainer.data.arrays()
        missing = [c for c in self.classes if not np.any(y == c)] if len(y) else list(self.classes)
        if missing:
            raise ValueError(f"no data for {', '.join(self.class_names.get(c, c) for c in missing)}")
        if progress:
            progress("fitting gestures", 0.3)
        clf = GestureClassifier(self.classes, self.trainer.clf.k)
        report = clf.fit(X, y)
        cmd_acc, confused = _command_accuracy(clf, X, y, self.command_map)
        levels = X[:, : self.channels]
        relaxed = np.median(levels[y == self.classes[0]], axis=0)
        strong = np.percentile(levels[y != self.classes[0]], 95, axis=0)
        return EmgCalibration(clf, report, cmd_acc, confused, relaxed, np.maximum(strong - relaxed, 1e-6))

    def apply_calibration(self, cal: "EmgCalibration", save: bool = True) -> "EmgCalibration":
        tr = self.trainer
        tr.clf, tr.report, tr.dirty = cal.clf, cal.report, False
        self.display_cal = (cal.relaxed, cal.span)
        tr.meta["display"] = {"relaxed": cal.relaxed.round(4).tolist(), "span": cal.span.round(4).tolist()}
        self.record_settle_s = config.TRAIN_SETTLE_S
        self._cal_backup = None
        if save:
            tr.save()
        return cal

    # -- pipeline ------------------------------------------------------------------------
    def _process(self) -> None:
        for t, sample in self._read_samples():
            for end, window in self.windower.push(t, sample):
                self._on_window(end, window)

    def _on_window(self, end: float, window: np.ndarray) -> None:
        self.windows += 1
        feats = extract(window, self.feature_names)
        level = feats[: self.channels]
        self._peak = max(self._peak * config.EMG_DISPLAY_PEAK_DECAY, float(np.max(level)))
        self.signal_full_scale = max(self.min_full_scale, self._peak)
        self._levels.append(np.asarray(level, dtype=float))
        self._chan_peak = np.maximum(self._chan_peak * config.EMG_DISPLAY_PEAK_DECAY, level)

        if self.recording is not None and end - self._record_since >= self.record_settle_s:
            self.trainer.add(feats, self.recording)
            self.recorded_this_hold += 1

        if self.trainer.clf.is_fitted:
            label, conf, probs = self.trainer.clf.predict(feats)
        else:
            label, conf, probs = None, 0.0, None
        pred = Prediction(label=label, confidence=conf, emg=level, timestamp=end, probs=probs)
        self._last = pred
        self._pending.append(pred)
        if len(self._pending) > _MAX_PENDING:
            del self._pending[: len(self._pending) - _MAX_PENDING]


@dataclass
class EmgCalibration:
    clf: GestureClassifier
    report: FitReport
    command_accuracy: Optional[float]
    confused: str            # the most common mix-up, e.g. "Left as Back 12%"
    relaxed: np.ndarray
    span: np.ndarray

    def describe(self) -> str:
        acc = f"{self.command_accuracy * 100:.0f}% right command" if self.command_accuracy is not None else "fitted"
        mix = f"; most mixed up: {self.confused}" if self.confused else ""
        return f"Gestures learned: {acc} on held-out data{mix}. Saved"


def _command_accuracy(clf: GestureClassifier, X: np.ndarray, y: np.ndarray,
                      command_map) -> tuple[Optional[float], str]:
    """Held-out accuracy at the command level (two classes that both mean S aren't an error)."""
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    _, counts = np.unique(y, return_counts=True)
    folds = min(config.TRAIN_CV_FOLDS, int(counts.min()))
    if folds < 2:
        return None, ""
    pred = cross_val_predict(clf._make(len(X)), X, y, cv=StratifiedKFold(n_splits=folds))
    want = np.array([command_map.get(c, "S") for c in y])
    got = np.array([command_map.get(c, "S") for c in pred])
    worst, worst_rate = "", 0.0
    for c in np.unique(want):
        mask = want == c
        for d in np.unique(got[mask]):
            rate = float(np.mean(got[mask] == d))
            if d != c and rate > worst_rate:
                worst, worst_rate = f"{commands.NAMES.get(c, c).title()} read as {commands.NAMES.get(d, d).title()} {rate * 100:.0f}%", rate
    return float(np.mean(want == got)), (worst if worst_rate >= 0.05 else "")
