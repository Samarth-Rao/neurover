"""Abstract input source.

A Source turns some input (keyboard, EMG armband, EEG headband...) into a
stream of Predictions. The controller polls ``latest()`` every frame and treats
a Prediction with a newer timestamp as one fresh "window" of data.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np


@dataclass(frozen=True, eq=False)
class Prediction:
    # Class name ("fist") or, for direct sources, a command letter ("F").
    # None means "no opinion" (e.g. no trained model yet) and never moves the rover.
    label: Optional[str]
    confidence: float
    # Latest per-channel signal level for the dashboard; None for non-signal sources.
    emg: Optional[np.ndarray]
    # time.monotonic() of the data this prediction was made from.
    timestamp: float
    # Per-class confidence for the dashboard bars (optional).
    probs: Optional[Mapping[str, float]] = None


@dataclass(frozen=True)
class SourceStatus:
    ok: bool
    message: str
    pending: bool = False  # not ready yet but working on it (e.g. scanning): shown yellow, not red


class Source(abc.ABC):
    #: Short name shown on the dashboard.
    name: str = "source"
    #: Labels this source can emit, in display order (dashboard confidence bars).
    classes: Sequence[str] = ()
    #: Maps class labels to commands. Direct sources emit command letters and leave this empty.
    command_map: Mapping[str, str] = {}
    #: Number of signal channels (0 = no live signal bars).
    channels: int = 0
    #: Whether training mode applies to this source.
    supports_training: bool = False
    #: Display names for class labels (dashboard). Falls back to config.GESTURE_NAMES.
    class_names: Mapping[str, str] = {}
    #: Override for the smoother's confidence threshold. Sources whose decoder already gates on its
    #: own calibrated threshold (EEG) set 0.0; None = use config.CONFIDENCE_THRESHOLD.
    confidence_threshold: Optional[float] = None

    @abc.abstractmethod
    def start(self) -> None:
        """Begin producing data. Must not block for long and must not raise for missing hardware;
        report hardware problems through ``status()`` instead."""

    @abc.abstractmethod
    def stop(self) -> None:
        """Release devices, threads and processes. Safe to call more than once."""

    @abc.abstractmethod
    def latest(self) -> Optional[Prediction]:
        """Most recent prediction, or None if nothing has been produced yet."""

    def poll(self) -> list[Prediction]:
        """Every prediction produced since the last poll, oldest first.

        The controller uses this so that no window is skipped when several arrive
        in one frame (e.g. a burst from the armband). Default: the latest one, if new.
        """
        p = self.latest()
        last = getattr(self, "_polled_ts", None)
        if p is None or (last is not None and p.timestamp <= last):
            return []
        self._polled_ts = p.timestamp
        return [p]

    def status(self) -> SourceStatus:
        return SourceStatus(True, "ready")

    def on_input(self, held: frozenset[str]) -> None:
        """Operator input from the UI: the set of held logical directions
        ("up", "down", "left", "right"). Sources that don't use it ignore it."""
