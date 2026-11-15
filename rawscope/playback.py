"""Play a recording back through the same view as the live feeds.

PlaybackDevice offers what the view reads from a live DeviceFeed (window / silent / rate, label,
positions, resistance...), cut at the playhead instead of "now". The 1-40 Hz trace is made with the
same causal filter as live, restarted at gaps, so a recording looks exactly as it did live.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

import config
from eeg.devices import GAP_RESET_S, SILENT_S, measured_rate, node_order
from eeg.filters import StreamFilter

from .analysis import runs

SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0)


class PlaybackDevice:
    def __init__(self, d: dict, fs: float) -> None:
        self.name = d["name"]
        self.fs = fs
        self.label = d.get("label", self.name)
        self.sdk = list(d["sdk"])
        self.positions = list(d["positions"])
        self.serial = d.get("serial") or None
        self.info = {"kind": d.get("kind", ""), "serial": self.serial}
        self.battery = d.get("battery")
        res = d.get("resistance", {})
        self.resistance = {c: res[p] for c, p in zip(self.sdk, self.positions) if p in res}
        self.state, self.message = "streaming", ""
        self.t = np.asarray(d["t"], dtype=float)
        self.x = np.asarray(d["x"], dtype=float).reshape(len(self.t), len(self.sdk))
        self.xf = np.zeros_like(self.x)
        filt = StreamFilter(fs, config.EEG_DISPLAY_BAND, config.EEG_NOTCH_HZ)
        for a, b in runs(self.t, GAP_RESET_S):
            filt.reset()
            self.xf[a:b] = filt.process(self.x[a:b])
        # Stretches with data (for the timeline): a gap is anything longer than a dropped packet or two.
        self.coverage = [(float(self.t[a]), float(self.t[b - 1])) for a, b in runs(self.t)]

    def window(self, t_end: float, seconds: float, kind: str = "raw") -> tuple[np.ndarray, np.ndarray]:
        """Like a live device's window(): "raw" µV or the 1-40 Hz "display" copy, cut at the playhead."""
        i0 = int(np.searchsorted(self.t, t_end - seconds, side="right"))
        i1 = int(np.searchsorted(self.t, t_end, side="right"))
        return self.t[i0:i1], (self.xf if kind == "display" else self.x)[i0:i1]

    def silent(self, t_end: float) -> bool:
        i = int(np.searchsorted(self.t, t_end, side="right"))
        return i == 0 or t_end - self.t[i - 1] > SILENT_S

    def rate(self, t_end: float) -> Optional[float]:
        return measured_rate(self.window(t_end, 3.0)[0])


class Playback:
    def __init__(self, rec: dict) -> None:
        self.rec = rec
        self.path = Path(rec["path"])
        self.meta = rec["meta"]
        self.fs = float(self.meta["fs"])
        self.devices = [PlaybackDevice(d, self.fs) for d in rec["devices"].values()]
        self.nodes = node_order(self.devices)
        firsts = [d.t[0] for d in self.devices if len(d.t)]
        lasts = [d.t[-1] for d in self.devices if len(d.t)]
        self.t0 = min([float(self.meta["start"]), *firsts])
        self.t1 = max([float(self.meta["end"]), *lasts])
        self.duration = max(0.0, self.t1 - self.t0)
        self.events = [(float(t), str(dev), str(text)) for t, dev, text in self.meta.get("events", [])]
        self.trials = [t for t in self.meta.get("trials") or [] if "cue" in t]   # cued trials, if any
        self.pos = 0.0          # seconds since the recording started
        self.speed_i = SPEEDS.index(1.0)
        self.playing = False
        self._last: Optional[float] = None

    @property
    def name(self) -> str:
        return self.meta.get("name") or "untitled"

    @property
    def speed(self) -> float:
        return SPEEDS[self.speed_i]

    @property
    def head(self) -> float:
        """The playhead on the recording's own clock (what the samples are timestamped with)."""
        return self.t0 + self.pos

    def update(self, now: float) -> None:
        if self.playing and self._last is not None:
            self.pos = min(self.duration, self.pos + (now - self._last) * self.speed)
            if self.pos >= self.duration:
                self.playing = False
        self._last = now

    def toggle(self, now: float) -> None:
        if not self.playing and self.pos >= self.duration:
            self.pos = 0.0                      # at the end: play again from the start
        self.playing = not self.playing
        self._last = now

    def seek(self, pos: float) -> None:
        self.pos = max(0.0, min(self.duration, pos))

    def cue_at(self, t: float) -> Optional[tuple[int, dict]]:
        """(trial number, counting from 1; 0 = practice) and the trial whose cue is showing at ``t``."""
        n = 0
        for tr in self.trials:
            n += 0 if tr.get("practice") else 1
            if tr["cue"] <= t < tr.get("end", tr["cue"]):
                return (0 if tr.get("practice") else n), tr
        return None

    def faster(self, step: int) -> None:
        self.speed_i = max(0, min(len(SPEEDS) - 1, self.speed_i + step))
