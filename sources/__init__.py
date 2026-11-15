"""Input sources. Each one turns an input (keys, EMG, EEG...) into Predictions.

Sources are imported lazily so that optional hardware libraries (pyomyo) are
only needed when that source is actually selected.
"""
from __future__ import annotations

from .base import Prediction, Source, SourceStatus

SOURCE_NAMES = ("keyboard", "mock", "myo", "eeg", "mock-eeg")


def create_source(name: str, **options) -> Source:
    if name == "keyboard":
        from .keyboard_source import KeyboardSource
        return KeyboardSource()
    if name == "mock":
        from .mock_emg_source import MockEmgSource
        return MockEmgSource(**options)
    if name == "myo":
        from .myo_source import MyoSource
        return MyoSource(**options)
    if name == "mock-eeg":
        from .mock_eeg_source import MockEegSource
        return MockEegSource(**options)
    if name == "eeg":
        from .eeg_source import BrainBitSource
        return BrainBitSource(**options)
    raise ValueError(f"unknown source {name!r}; choose from {', '.join(SOURCE_NAMES)}")


__all__ = ["Prediction", "Source", "SourceStatus", "SOURCE_NAMES", "create_source"]
