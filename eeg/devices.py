"""Live EEG devices, shared by the rover and the scope.

One BrainBit SDK thread per device (eeg/brainbit.py), or a mock. Each device keeps its own samples on
its own clock exactly as the worker timestamps them: nothing is resampled or merged, so a device that
drops out simply has no recent samples, and the other keeps going. Raw values are µV as the SDK
reports them; each device also keeps causally filtered copies (``BANDS``) for display, decoding and the muscle check.
"""
from __future__ import annotations

import importlib.util
import math
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

import config

from .filters import StreamFilter
from .recording import Recorder

BUFFER_S = 30.0            # history kept per device
GAP_RESET_S = 0.5          # a gap this long restarts the filters (new DC offset after a reconnect)
SILENT_S = 1.0             # no samples for this long = the device is silent
BANDS = {"display": config.EEG_DISPLAY_BAND, "decode": config.SSVEP_BAND, "muscle": config.MOVE_MUSCLE_BAND}
_MAX_MSGS_PER_POLL = 4000
_STATE_ORDER = ("streaming", "resist", "connecting", "reconnecting", "scanning", "starting", "error", "idle")


@dataclass(frozen=True)
class Node:
    """One electrode: which device, which SDK channel, where it sits on the head."""
    device: str
    label: str      # device label, e.g. "Flex"
    sdk: str        # SDK channel name, e.g. "O1"
    position: str   # physical position, e.g. "PO3"
    col: int        # column in that device's sample array


class Ring:
    """The last ``n`` samples of a timestamped multichannel stream (amortised O(1) appends)."""

    def __init__(self, width: int, n: int) -> None:
        self.n = max(1, int(n))
        self.t = np.zeros(2 * self.n)
        self.x = np.zeros((2 * self.n, width))
        self.len = 0

    def push(self, t: np.ndarray, x: np.ndarray) -> None:
        m = len(t)
        if m == 0:
            return
        if m >= self.n:
            t, x, m = t[-self.n:], x[-self.n:], self.n
        if self.len + m > 2 * self.n:
            keep = self.n - m
            self.t[:keep] = self.t[self.len - keep:self.len]
            self.x[:keep] = self.x[self.len - keep:self.len]
            self.len = keep
        self.t[self.len:self.len + m] = t
        self.x[self.len:self.len + m] = x
        self.len += m

    def view(self, seconds: Optional[float] = None, t_end: Optional[float] = None) -> tuple[np.ndarray, np.ndarray]:
        """Samples in (t_end - seconds, t_end] (defaults: everything / up to the newest)."""
        t, x = self.t[:self.len], self.x[:self.len]
        if not self.len:
            return t, x
        i1 = self.len if t_end is None else int(np.searchsorted(t, t_end, side="right"))
        if seconds is None:
            return t[:i1], x[:i1]
        end = t[i1 - 1] if t_end is None and i1 else (t_end if t_end is not None else 0.0)
        i0 = int(np.searchsorted(t, end - seconds, side="right"))
        return t[i0:i1], x[i0:i1]


class DeviceFeed:
    """One headset: its worker, queue, buffers and latest status."""

    def __init__(self, name: str, spec: Mapping, fs: float, positions: Optional[Mapping[str, str]] = None) -> None:
        self.name = name
        self.label = spec.get("label", name)
        self.family = spec.get("family", "any")
        self.serial = spec.get("serial")
        self.sdk = list(spec["map"])                     # SDK channel names, in the order samples arrive
        self.config_positions = [spec["map"][c] for c in self.sdk]
        self.positions = [positions[c] for c in self.sdk] if positions else list(self.config_positions)
        self.fs = fs
        self.q: "queue.Queue" = queue.Queue()
        self.link = None
        self.state = "idle"
        self.message = ""
        self.info: dict = {}
        self.battery: Optional[int] = None
        self.resistance: dict[str, float] = {}           # by SDK channel name, ohms
        n = BUFFER_S * fs
        self.rings = {"raw": Ring(len(self.sdk), n), **{b: Ring(len(self.sdk), n) for b in BANDS}}
        self._filters = {b: StreamFilter(fs, band, config.EEG_NOTCH_HZ) for b, band in BANDS.items()}
        self.last_arrival: Optional[float] = None        # our clock, when samples last arrived
        self.samples = 0

    def push(self, t: np.ndarray, x: np.ndarray, now: float) -> None:
        if len(t) == 0:
            return
        prev = self.rings["raw"].view()[0]
        if len(prev) and t[0] - prev[-1] > GAP_RESET_S:
            for f in self._filters.values():
                f.reset()
        self.rings["raw"].push(t, x)
        for b, f in self._filters.items():
            self.rings[b].push(t, f.process(x))
        self.last_arrival = now
        self.samples += len(t)

    def window(self, t_end: Optional[float], seconds: float, kind: str = "raw") -> tuple[np.ndarray, np.ndarray]:
        """The last ``seconds`` of samples up to ``t_end`` (None = the newest): raw µV, "display"
        (1-40 Hz) or "decode" (the SSVEP band)."""
        return self.rings[kind].view(seconds, t_end)

    def silent(self, now: float) -> bool:
        return self.last_arrival is None or now - self.last_arrival > SILENT_S

    def rate(self, t_end: Optional[float] = None) -> Optional[float]:
        return measured_rate(self.rings["raw"].view(3.0)[0])

    def snapshot(self) -> dict:
        """What a recording keeps about this device."""
        return {"name": self.name, "label": self.label, "kind": self.info.get("kind", ""),
                "serial": str(self.info.get("serial") or self.serial or ""), "sdk": list(self.sdk),
                "positions": list(self.positions), "battery": self.battery,
                "resistance": {self.positions[i]: self.resistance[c] for i, c in enumerate(self.sdk)
                               if c in self.resistance}}


def measured_rate(t: np.ndarray) -> Optional[float]:
    """Sample rate (Hz) from timestamps."""
    if len(t) < 10 or t[-1] <= t[0]:
        return None
    return (len(t) - 1) / (t[-1] - t[0])


def node_order(devices: Sequence) -> list[Node]:
    """Device by device; within a device, in config.EEG_DISPLAY_ORDER."""
    rank = {p: i for i, p in enumerate(config.EEG_DISPLAY_ORDER)}
    out = []
    for d in devices:
        cols = sorted(range(len(d.sdk)), key=lambda i: (rank.get(d.positions[i], len(rank)), i))
        out += [Node(d.name, d.label, d.sdk[i], d.positions[i], i) for i in cols]
    return out


def reassign(positions: Mapping[str, list[str]], device: str, index: int, position: str,
             labels: Optional[Mapping[str, str]] = None) -> dict[str, list[str]]:
    """Every device's positions after electrode ``index`` of ``device`` moves to ``position``. Taking
    the spot of another electrode of the same device swaps the two; a spot another device has is
    refused (ValueError): two electrodes can't be in one place."""
    position = position.strip()
    if not position:
        raise ValueError("an electrode needs a position name")
    out = {k: list(v) for k, v in positions.items()}
    for other, ps in out.items():
        if other != device and position in ps:
            raise ValueError(f"{position} is taken by the {(labels or {}).get(other, other)}")
    mine = out[device]
    if position in mine:
        mine[mine.index(position)] = mine[index]
    mine[index] = position
    return out


class Feeds:
    def __init__(self, device_names: Sequence[str] = config.EEG_USE_DEVICES,
                 devices: Mapping[str, Mapping] = config.EEG_DEVICES, *, mock: bool = False,
                 mock_preset: str = "realistic", mock_motor: bool = False, fs: float = config.EEG_SAMPLE_HZ,
                 clock: Callable[[], float] = time.monotonic, seed: Optional[int] = None,
                 placement: Optional[Mapping[str, Mapping[str, str]]] = None) -> None:
        unknown = [n for n in device_names if n not in devices]
        if unknown:
            raise ValueError(f"unknown device(s) {unknown}; config.EEG_DEVICES has {', '.join(devices)}")
        if not device_names:
            raise ValueError("no devices selected")
        self.fs = fs
        self.clock = clock
        self.mock = mock
        self.placement_note = ""
        usable = self._usable_placement(device_names, devices, placement or {})
        self.devices = [DeviceFeed(n, devices[n], fs, usable.get(n)) for n in device_names]
        self.nodes = node_order(self.devices)
        self.recorder: Optional[Recorder] = None
        self.fatal: Optional[str] = None
        self._mock: Optional[_MockLinks] = (_MockLinks(self.devices, fs, clock, seed, mock_preset, mock_motor)
                                            if mock else None)

    def device(self, name: str) -> DeviceFeed:
        return next(d for d in self.devices if d.name == name)

    # -- lifecycle ---------------------------------------------------------------------
    def start(self) -> None:
        if self._mock is not None:
            self._mock.start()
            return
        if importlib.util.find_spec("neurosdk") is None:
            self.fatal = "pyneurosdk2 is not installed. Run: pip install pyneurosdk2"
            return
        from .brainbit import BrainBitLink

        single = len(self.devices) == 1
        for d in self.devices:
            # Measuring contact while the other device connects can leave the Flex silent (see
            # brainbit._BRINGUP), so with two devices contact is measured on request (R) only.
            d.link = BrainBitLink(d.q, d.sdk, None, self.fs, family=d.family, label=d.label,
                                  serial=d.serial, measure_resist=single)
            d.link.start()
            d.state, d.message = "starting", f"starting {d.label}…"

    def stop(self) -> None:
        links = [d.link for d in self.devices if d.link is not None]
        for d in self.devices:
            d.link = None
        for link in links:          # tell them all first so they shut down in parallel
            link.stop_event.set()
        for link in links:
            link.join(timeout=5.0)

    # -- data --------------------------------------------------------------------------
    def poll(self) -> None:
        now = self.clock()
        if self._mock is not None:
            self._mock.pump(now)
        for d in self.devices:
            self._drain(d, now)

    def _drain(self, d: DeviceFeed, now: float) -> None:
        for _ in range(_MAX_MSGS_PER_POLL):
            try:
                msg = d.q.get_nowait()
            except queue.Empty:
                return
            kind = msg[0]
            if kind == "eeg":
                t, x = np.asarray(msg[1], dtype=float), np.asarray(msg[2], dtype=float).reshape(-1, len(d.sdk))
                d.push(t, x, now)
                if self.recorder is not None:
                    self.recorder.add(d.name, t, x)
            elif kind == "status":
                if msg[1] != d.state or msg[1] == "error":   # not every repeat of "still scanning (n s)…"
                    print(f"[eeg] {d.label}: {msg[1]} - {msg[2]}", flush=True)
                    if self.recorder is not None:
                        self.recorder.event(now, d.name, f"{msg[1]}: {msg[2]}")
                d.state, d.message = msg[1], msg[2]
            elif kind == "device":
                d.info = msg[1]
                d.battery = msg[1].get("battery")
                print(f"[eeg] {d.label}: {msg[1].get('kind')} {msg[1].get('name')} serial "
                      f"{msg[1].get('serial')}, channels {msg[1].get('channels')}, battery {d.battery}%", flush=True)
            elif kind == "battery":
                d.battery = msg[1]
            elif kind == "resist":
                d.resistance.update(msg[1])
                if self.recorder is not None:
                    self.recorder.event(now, d.name, "resistance " + ", ".join(
                        f"{d.positions[d.sdk.index(c)]}={ohm_text(r)}" for c, r in msg[1].items() if c in d.sdk))

    # -- contact -----------------------------------------------------------------------
    def measure_contact(self) -> bool:
        """Electrode resistance on every streaming device (R). They pause, are measured one at a time
        (~5 s each), then resume."""
        if self._mock is not None:
            return self._mock.measure_contact()
        ready = [d for d in self.devices if d.link is not None and d.state == "streaming"]
        if not ready:
            return False
        rnd = threading.Barrier(len(ready))
        for d in ready:
            d.link.resist_round = rnd
            d.link.resist_request.set()
        return True

    # -- where the electrodes are ---------------------------------------------------------
    def _usable_placement(self, names, devices, placement) -> dict[str, dict[str, str]]:
        """A saved placement for each device whose SDK channels it covers exactly; ignored (with a
        note) if it would put two electrodes in one spot."""
        use = {n: dict(placement[n]) for n in names if n in placement and set(placement[n]) == set(devices[n]["map"])}
        final = [p for n in names for p in (use[n].values() if n in use else devices[n]["map"].values())]
        if len(set(final)) != len(final):
            self.placement_note = "saved placement ignored: it puts two electrodes in one spot"
            print(f"[eeg] {self.placement_note}", flush=True)
            return {}
        return use

    def placement(self) -> dict[str, dict[str, str]]:
        """device -> {SDK channel: position}, as it is now."""
        return {d.name: dict(zip(d.sdk, d.positions)) for d in self.devices}

    def set_position(self, device: str, sdk: str, position: str) -> None:
        """Move one electrode (swaps with another of the same device that was there; ValueError if another
        device has that spot, or while recording: the recording would be half one way, half the other)."""
        if self.recording:
            raise ValueError("stop recording before moving an electrode")
        d = self.device(device)
        new = reassign({x.name: x.positions for x in self.devices}, device, d.sdk.index(sdk), position,
                       {x.name: x.label for x in self.devices})
        for x in self.devices:
            x.positions = new[x.name]
        # Rows stay where they are on screen; only their names change.
        self.nodes = [Node(n.device, n.label, n.sdk, new[n.device][n.col], n.col) for n in self.nodes]

    # -- recording ---------------------------------------------------------------------
    @property
    def recording(self) -> bool:
        return self.recorder is not None

    def start_recording(self) -> None:
        self.recorder = Recorder(self.clock(), self.fs, mock=self.mock)

    def stop_recording(self) -> Optional[dict]:
        """The finished recording (see recording.py), or None if nothing was recording."""
        rec, self.recorder = self.recorder, None
        if rec is None:
            return None
        return rec.finish(self.clock(), [d.snapshot() for d in self.devices])

    # -- aggregate state ---------------------------------------------------------------
    @property
    def state(self) -> str:
        if self.fatal:
            return "error"
        states = {d.state for d in self.devices}
        return next(s for s in _STATE_ORDER if s in states)

    def mock_input(self, gaze: Optional[str] = None, jaw: bool = False) -> None:
        """Mock: which flicker box the virtual user looks at, and whether they clench their jaw."""
        if self._mock is not None:
            self._mock.head.gaze = gaze
            self._mock.head.jaw = jaw

    def mock_cue(self, hand: Optional[str]) -> None:
        """Mock (with mock_motor): which hand the virtual subject moves / imagines ("left", "right", None)."""
        if self._mock is not None:
            self._mock.head.hand = hand


PLACEMENT_PATH = config.ROOT_DIR / "scope_placement.json"


def load_placement(path=PLACEMENT_PATH) -> dict[str, dict[str, str]]:
    """Where the electrodes are, as last set by clicking names in the scope: device -> {SDK channel:
    position}. {} if there's no (readable) file; the devices then use config.EEG_DEVICES."""
    import json

    try:
        data = json.loads(open(path).read())
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(d): {str(k): str(v) for k, v in m.items()} for d, m in data.items() if isinstance(m, dict)}


def save_placement(maps: Mapping[str, Mapping[str, str]], path=PLACEMENT_PATH) -> None:
    import json
    from pathlib import Path

    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({d: dict(m) for d, m in maps.items()}, indent=2) + "\n")
    tmp.replace(path)


def ohm_text(r: float) -> str:
    if not math.isfinite(r):
        return "open"
    return f"{r / 1e6:.1f}MΩ" if r >= 1e5 else f"{r / 1e3:.0f}kΩ"


class _MockLinks:
    """Synthetic devices: one virtual head (eeg/mock.py) over every device's positions, each device
    handed its own columns through its queue, like a real worker would."""

    def __init__(self, devices: Sequence[DeviceFeed], fs: float, clock: Callable[[], float],
                 seed: Optional[int], preset: str, motor: bool = False) -> None:
        from .mock import PRESETS, MockHead
        from .ssvep import target_frequencies

        self.devices = list(devices)
        self.fs = fs
        self.clock = clock
        positions = [p for d in self.devices for p in d.positions]
        device_of = [k for k, d in enumerate(self.devices) for _ in d.positions]
        self.head = MockHead(positions, fs, target_frequencies(), device_of, seed=seed, motor=motor, **PRESETS[preset])
        self.rng = np.random.default_rng(seed)
        self._cols, k = [], 0
        for d in self.devices:
            self._cols.append(slice(k, k + len(d.positions)))
            k += len(d.positions)
        self._t: Optional[float] = None
        self._resist_until: Optional[float] = None

    def start(self) -> None:
        self._t = self.clock()
        for d in self.devices:
            d.q.put(("device", {"kind": f"mock {d.label}", "name": f"Mock {d.label}", "serial": "mock",
                                "channels": list(d.sdk), "battery": 100}))
            d.q.put(("status", "streaming", "streaming (mock)"))

    def measure_contact(self) -> bool:
        if self._t is None:
            return False
        self._resist_until = self.clock() + 1.0
        for d in self.devices:
            d.q.put(("status", "resist", "measuring electrode contact… (mock)"))
        return True

    def pump(self, now: float) -> None:
        if self._t is None:
            return
        if self._resist_until is not None:
            if now < self._resist_until:
                self._t = now                   # no signal while measuring, like the real devices
                return
            self._resist_until = None
            for d in self.devices:
                d.q.put(("resist", {c: float(self.rng.uniform(150e3, 2.5e6)) for c in d.sdk}))
                d.q.put(("status", "streaming", "streaming (mock)"))
        if now - self._t > 1.0:
            self._t = now - 1.0
        n = int((now - self._t) * self.fs + 1e-9)
        if n <= 0:
            return
        t = self._t + np.arange(1, n + 1) / self.fs
        self._t = float(t[-1])
        x = self.head.generate(n)
        for d, cols in zip(self.devices, self._cols):
            d.q.put(("eeg", t, x[:, cols]))
