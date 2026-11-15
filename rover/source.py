"""Input sources: something that turns an input into Predictions the controller can smooth.

    KeyboardSource   WASD / arrows held = F/B/L/R, nothing = S (no signal; for trying the sim)
    EegSource        BrainBit Flex + headband (or the mock): SSVEP flicker boxes, eeg/ssvep.py

Every ``hop`` the EEG source takes each streaming device's latest window, lets eeg/clean.py judge
its electrodes and weight its samples, scores it, pools the devices and decides. A device that is
silent or has no usable electrode just isn't in that window; if no device is, there is no
prediction at all and the controller's dead-man stops the rover. While the driver moves or clenches
(eeg/ssvep.py StopGate) the prediction is a forced stop: S at once, whatever the boxes score.
"""
from __future__ import annotations

import abc
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

import commands
import config
from eeg import cues, recording, ssvep
from eeg.clean import FLAT, NOISY, SATURATED
from eeg.devices import Feeds, ohm_text
from eeg.filters import spectrum

DIRECTION_TO_COMMAND = {"up": commands.FORWARD, "down": commands.BACK, "left": commands.LEFT,
                        "right": commands.RIGHT}


@dataclass(frozen=True, eq=False)
class Prediction:
    label: Optional[str]          # a class (EEG: a box or "rest") or a command letter (keyboard)
    confidence: float
    timestamp: float              # time.monotonic() of the data it was made from
    scores: Optional[Mapping[str, float]] = None   # per class, for the dashboard
    forced: bool = False          # adopt at once, no debounce (the EEG stop gate)


@dataclass(frozen=True)
class SourceStatus:
    ok: bool
    message: str
    pending: bool = False         # not ready / degraded but working on it: yellow, not red


class Source(abc.ABC):
    name: str = "source"
    classes: Sequence[str] = ()
    command_map: Mapping[str, str] = {}      # class -> command (direct sources emit commands)
    class_names: Mapping[str, str] = {}
    confidence_threshold: Optional[float] = None   # None = config.CONFIDENCE_THRESHOLD
    eeg = False

    @abc.abstractmethod
    def start(self) -> None:
        """Begin. Must not block or raise for missing hardware: report it through status()."""

    @abc.abstractmethod
    def stop(self) -> None:
        """Release devices and threads. Safe to call twice."""

    @abc.abstractmethod
    def poll(self) -> list[Prediction]:
        """Every prediction made since the last call, oldest first."""

    def status(self) -> SourceStatus:
        return SourceStatus(True, "ready")

    def on_input(self, held: frozenset[str]) -> None:
        """Held keys from the UI ("up", "down", "left", "right", "jaw"); most sources ignore them."""


class KeyboardSource(Source):
    name = "keyboard"
    classes = commands.ALL

    def __init__(self, period_s: float = config.KEYBOARD_PERIOD_S, clock: Callable[[], float] = time.monotonic) -> None:
        self._period = period_s
        self._clock = clock
        self._order: list[str] = []      # held directions, oldest first; the newest one wins
        self._running = False
        self._last_t: Optional[float] = None

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def status(self) -> SourceStatus:
        return SourceStatus(self._running, "WASD / arrow keys" if self._running else "stopped")

    def on_input(self, held: frozenset[str]) -> None:
        held = {d for d in held if d in DIRECTION_TO_COMMAND}
        kept = [d for d in self._order if d in held]
        self._order = kept + sorted(held.difference(kept))

    def current_command(self) -> str:
        return DIRECTION_TO_COMMAND[self._order[-1]] if self._order else commands.STOP

    def poll(self) -> list[Prediction]:
        """One prediction per period, so the debounce feels the same as for EEG."""
        now = self._clock()
        if not self._running or (self._last_t is not None and now - self._last_t < self._period):
            return []
        self._last_t = now
        cmd = self.current_command()
        return [Prediction(cmd, 1.0, now, {c: float(c == cmd) for c in commands.ALL})]


def _setup(electrodes: str, view: str) -> str:
    return electrodes if view == "sim" else f"{electrodes} | view:{view}"


class EegSource(Source):
    eeg = True
    # The decoder gates on its own threshold + margin; the smoother only debounces (and the dead-man).
    confidence_threshold = float("-inf")

    def __init__(self, feeds: Feeds, *, user: str = "default", models_dir: Path = config.MODELS_DIR,
                 recordings_dir: Path = recording.RECORDINGS_DIR, window_s: Optional[float] = None,
                 view: str = "sim", clock: Callable[[], float] = time.monotonic) -> None:
        self.feeds = feeds
        self.view = view          # what's in the middle of the screen: "sim" or "camera" (part of the setup)
        self.name = "mock" if feeds.mock else "eeg"
        self.fs = feeds.fs
        self.user = user
        self.models_dir = Path(models_dir)
        self.recordings_dir = Path(recordings_dir)
        self.clock = clock
        self.targets = ssvep.target_frequencies()
        self.labels = list(self.targets)
        self.classes = (*self.labels, ssvep.REST)
        self.command_map = dict(config.SSVEP_COMMANDS)
        self.class_names = {l: f"{ssvep.ARROWS[l]} {f:.1f} Hz" for l, f in self.targets.items()}
        self.class_names[ssvep.REST] = "Rest"
        scorer = ssvep.Scorer(self.targets)
        self.decoders = {d.name: ssvep.DeviceDecoder(len(d.sdk), scorer) for d in feeds.devices}
        self.gate = ssvep.StopGate()
        self.stopped: Optional[str] = None             # why the stop gate holds (MOVING / MUSCLE), or None
        saved, self.load_message = ssvep.load_settings(user, self.setup(), self.models_dir)
        self.settings = saved or ssvep.Settings(user=user)
        if window_s is not None:
            if not 1.0 <= window_s <= 4.0:
                raise ValueError("the SSVEP window must be 1-4 s")
            self.settings.window_s = window_s
        self.scores = {c: 0.0 for c in self.classes}
        self.score_history: deque = deque(maxlen=int(12 / config.SSVEP_HOP_S))
        self.decoding: dict[str, list[str]] = {}       # device -> positions decoded in the latest window
        self.spectrum: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._next_hop: Optional[float] = None
        self._pending: list[Prediction] = []
        self._hops = 0

    # -- what the controller / dashboard read ------------------------------------------------
    @property
    def smooth_windows(self) -> int:
        return int(self.settings.n_smooth)

    @property
    def threshold(self) -> float:
        return float(self.settings.threshold)

    def setup(self) -> str:
        """The electrodes, and what's in the middle of the screen (a calibration made looking at the
        simulated arena isn't used with the live camera, and the other way round)."""
        return _setup(ssvep.setup_key({d.name: d.positions for d in self.feeds.devices}), self.view)

    def electrodes(self) -> list[dict]:
        """One row per electrode for the dashboard: position, device, status, level (µV), resistance, used."""
        out = []
        for nd in self.feeds.nodes:
            d = self.feeds.device(nd.device)
            cl = self.decoders[nd.device].cleaner
            silent = d.silent(self.clock())
            out.append({"position": nd.position, "device": d.label, "sdk": nd.sdk, "col": nd.col, "node": nd,
                        "status": "no data" if silent else cl.status[nd.col],
                        "level": None if silent or not np.isfinite(cl.level[nd.col]) else float(cl.level[nd.col]),
                        "resistance": d.resistance.get(nd.sdk),
                        "used": nd.position in self.decoding.get(nd.device, ())})
        return out

    def left_out(self) -> list[str]:
        """Electrodes of streaming devices that sit out of decoding, with why."""
        out = []
        for row in self.electrodes():
            if row["status"] in (NOISY, FLAT, SATURATED):
                out.append(f"{row['position']} {row['status']}")
        return out

    # -- lifecycle -----------------------------------------------------------------------------
    def start(self) -> None:
        self.feeds.start()

    def stop(self) -> None:
        if self.feeds.recording:
            self.feeds.stop_recording()
        self.feeds.stop()

    def on_input(self, held: frozenset[str]) -> None:
        if self.feeds.mock:
            dirs = [d for d in ("up", "down", "left", "right") if d in held]
            self.feeds.mock_input(gaze=dirs[-1] if dirs else None, jaw="jaw" in held)

    def measure_resistance(self) -> bool:
        return self.feeds.measure_contact()

    # -- decoding ------------------------------------------------------------------------------
    def poll(self) -> list[Prediction]:
        self.feeds.poll()
        now = self.clock()
        if self._next_hop is None:
            self._next_hop = now
        if now >= self._next_hop:
            self._decide(now)
            self._next_hop += config.SSVEP_HOP_S
            if self._next_hop <= now:           # fell behind (UI stall): don't burst, just resync
                self._next_hop = now + config.SSVEP_HOP_S
        out, self._pending = self._pending, []
        return out

    def _decide(self, now: float) -> None:
        w_s = self.settings.window_s
        zs, why, decoding, spec = [], [], {}, []
        for d in self.feeds.devices:
            dec = self.decoders[d.name]
            dec.z = None
            if d.silent(now):
                continue
            t, x = d.window(None, w_s, "decode")
            _, raw = d.window(None, w_s, "raw")
            _, muscle = d.window(None, w_s, "muscle")
            z = dec.step(t, x, raw, self.fs, w_s, muscle)
            if z is None:
                continue
            zs.append(z)
            why.append(dec.disturbed)
            decoding[d.name] = [d.positions[i] for i in dec.used]
            if self._hops % 5 == 0:
                spec.append(spectrum(x[:, dec.used].mean(axis=1), self.fs))
        self.decoding = decoding
        self._hops += 1
        if spec:
            freqs = spec[0][0]
            psd = np.mean([p for f, p in spec if len(p) == len(freqs)], axis=0)
            if self.spectrum is not None and len(self.spectrum[1]) == len(psd):
                psd = 0.6 * self.spectrum[1] + 0.4 * psd
            self.spectrum = (freqs, psd)
        self.stopped = self.gate.update(now, why)
        if not zs:
            return                               # no prediction: the dead-man stops the rover
        z = ssvep.pool(zs)
        i = ssvep.decide(z, self.settings.threshold, self.settings.margin)
        label = self.labels[i] if i >= 0 and not self.stopped else ssvep.REST
        self.scores = {**dict(zip(self.labels, map(float, z))), ssvep.REST: 0.0}
        self.score_history.append((now, dict(self.scores), label))
        self._pending.append(Prediction(label, float(z.max()), now, dict(self.scores), forced=bool(self.stopped)))
        del self._pending[:-200]

    # -- status --------------------------------------------------------------------------------
    def status(self) -> SourceStatus:
        f = self.feeds
        if f.fatal:
            return SourceStatus(False, f.fatal)
        now = self.clock()
        live = [d for d in f.devices if d.state == "streaming" and not d.silent(now)]
        if not live:
            if all(d.state == "error" for d in f.devices):
                return SourceStatus(False, " · ".join(f"{d.label}: {d.message}" for d in f.devices) + " Retrying…")
            return SourceStatus(False, " · ".join(f"{d.label}: {d.message or d.state}" for d in f.devices),
                                pending=True)
        bat = [f"{d.label} {d.battery}%" for d in live if d.battery is not None]
        msg = " + ".join(d.label for d in live) + " streaming" + (f" (battery {', '.join(bat)})" if bat else "")
        notes = []
        missing = [d for d in f.devices if d not in live]
        if missing:
            notes.append(" · ".join(f"{d.label}: {d.message or d.state}" for d in missing) + " (decoding without it)")
        out = self.left_out()
        if out:
            notes.append(f"left out: {', '.join(out)} (reseat; R checks contact)")
        poor = [f"{f.device(n.device).label} {n.position} {ohm_text(r)}" for n in f.nodes
                for r in [f.device(n.device).resistance.get(n.sdk)] if r is not None and not r < config.EEG_CONTACT_GOOD_OHM]
        if poor:
            notes.append(f"high resistance: {', '.join(poor)}")
        if notes:
            return SourceStatus(False, msg + "; " + "; ".join(notes), pending=True)
        return SourceStatus(True, msg)

    # -- calibration (C) -----------------------------------------------------------------------
    def calibration_protocol(self) -> cues.Protocol:
        return ssvep.calibration_protocol(self.targets)

    def start_calibration(self) -> None:
        self.feeds.start_recording()

    def cancel_calibration(self) -> None:
        self.feeds.stop_recording()

    def finish_calibration(self, meta: Optional[dict] = None) -> tuple[Optional[dict], Optional[Path]]:
        """Stop recording and save it (a normal recording, so the scope can play it back). ``meta``: the
        cue run's trials + protocol, and the flicker timing."""
        rec = self.feeds.stop_recording()
        if rec is None or not any(len(d["t"]) for d in rec["devices"].values()):
            return None, None
        rec["meta"].update(kind="ssvep-calibration", user=self.user, frames=dict(config.SSVEP_FRAMES), view=self.view,
                           **(meta or {}))
        try:
            path = recording.save(rec, self.recordings_dir, f"ssvep-calibration-{self.user}")
        except Exception as exc:
            print(f"[rover] couldn't save the calibration recording: {exc}")
            path = None
        return rec, path

    def earlier_calibrations(self, exclude: Optional[Path] = None) -> list[Path]:
        """This user's saved calibrations made with the electrodes as they are now, the same flicker
        frequencies, resting the same way (at the rover or the centre), newest first. Change any of those
        and calibrating starts from zero."""
        out = []
        skip = Path(exclude).resolve() if exclude else None
        for meta in recording.list_recordings(self.recordings_dir):
            if meta.get("kind") != "ssvep-calibration" or meta.get("user") != self.user or meta["path"].resolve() == skip:
                continue
            key = ssvep.setup_key({d["name"]: d["positions"] for d in meta.get("devices", [])})
            if _setup(key, meta.get("view", "sim")) == self.setup() and meta.get("frames") == dict(config.SSVEP_FRAMES) \
                    and ssvep.rest_style(meta) == ssvep.rest_style({"protocol": self.calibration_protocol().to_meta()}):
                out.append(meta["path"])
        return out

    def fit_calibration(self, rec: dict, progress=None) -> ssvep.Settings:
        """Fit on this calibration plus the earlier ones with the same electrodes (up to
        SSVEP_CAL_COMBINE in all), so calibrating again adds data instead of starting over. Safe to run
        in a worker thread: changes nothing live."""
        path = rec.get("path")
        earlier = self.earlier_calibrations(exclude=path)[:max(0, config.SSVEP_CAL_COMBINE - 1)]
        recs = [rec] + [recording.load(p) for p in earlier]
        s = ssvep.fit(recs, progress=progress)
        s.setup, s.user = self.setup(), self.user
        s.recording = ", ".join(Path(r["path"]).name for r in recs if r.get("path"))
        return s

    def apply_calibration(self, s: ssvep.Settings) -> ssvep.Settings:
        self.settings = s
        ssvep.save_settings(s, self.models_dir)
        return s
