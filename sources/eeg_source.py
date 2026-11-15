"""BrainBit EEG source (--source eeg), via BrainBit's pyneurosdk2: a BrainBit Flex, the
BrainBit headband, or both at once (config.EEG_DEVICES / EEG_USE_DEVICES, --eeg-devices).

Each device's SDK runs in its own thread (sources/brainbit_worker.py); its callbacks push
samples into a thread-safe queue that this class drains every frame, so the UI never
blocks. With two devices the streams are put on one timeline (sources/eeg_merge.py). The
shared EEG pipeline (eeg_common.py) does the rest: channel mapping, filtering, artifact
rejection, SSVEP / alpha decoding.

Before streaming, electrode resistance is measured and shown per position on the
dashboard (press R to measure again). Streaming starts regardless, with a warning
for poor contacts. If Bluetooth drops, the worker reconnects; meanwhile that device's
electrodes are left out of decoding (with no device left, no data arrives and the
dead-man forces S).
"""
from __future__ import annotations

import importlib.util
import math
import queue
import threading
import time
from typing import Callable, Optional

import numpy as np

import config

from .base import SourceStatus
from .brainbit_worker import BrainBitLink
from .eeg_common import EegStreamSource
from .eeg_merge import StreamMerger

_MAX_CHUNKS_PER_POLL = 2000
# Most useful first: what the source as a whole reports when no device is streaming yet.
_STATE_ORDER = ("streaming", "resist", "connecting", "reconnecting", "scanning", "starting", "error", "idle")


class _Device:
    """One headset: its worker thread, queue and latest status."""

    def __init__(self, name: str, spec: dict) -> None:
        self.name = name
        self.label = spec.get("label", name)
        self.family = spec.get("family", "any")
        self.sdk = tuple(spec["map"])
        self.serial = spec.get("serial")
        self.q: "queue.Queue" = queue.Queue()
        self.link: Optional[BrainBitLink] = None
        self.state = "idle"
        self.message = ""
        self.info: dict = {}
        self.battery: Optional[int] = None


class BrainBitSource(EegStreamSource):
    name = "eeg"

    def __init__(self, mode: str = "ssvep", *, device: Optional[str] = config.BRAINBIT_DEVICE,
                 clock: Callable[[], float] = time.monotonic, **kwargs) -> None:
        super().__init__(mode, clock=clock, **kwargs)
        self.device = device  # name/serial filter; only used with a single device
        self.links = [_Device(n, self.devices[n]) for n in self.device_names]
        self._merger = (StreamMerger([len(d.sdk) for d in self.links], self.fs, config.EEG_MERGE_LAG_S,
                                     config.EEG_MERGE_MAX_GAP_S) if len(self.links) > 1 else None)
        self.resistance: dict[str, float] = {}   # by physical position, ohms
        self._fatal: Optional[str] = None

    # -- aggregate state -----------------------------------------------------------------
    @property
    def state(self) -> str:
        if self._fatal:
            return "error"
        states = {d.state for d in self.links}
        return next(s for s in _STATE_ORDER if s in states)

    @property
    def message(self) -> str:
        if self._fatal:
            return self._fatal
        if len(self.links) == 1:
            return self.links[0].message
        return " · ".join(f"{d.label}: {d.message or d.state}" for d in self.links)

    @property
    def device_info(self) -> dict:
        if len(self.links) == 1:
            return self.links[0].info
        kinds = [d.info["kind"] for d in self.links if d.info.get("kind")]
        return {"kind": " + ".join(kinds)} if kinds else {}

    @property
    def battery(self) -> Optional[int]:
        levels = [d.battery for d in self.links if d.battery is not None]
        return min(levels) if levels else None

    # -- lifecycle ---------------------------------------------------------------------
    def start(self) -> None:
        if importlib.util.find_spec("neurosdk") is None:
            self._fatal = "pyneurosdk2 is not installed. Run: pip install pyneurosdk2"
            return
        single = len(self.links) == 1
        for d in self.links:
            d.link = BrainBitLink(d.q, d.sdk, self.device if single else None, self.fs,
                                  family=d.family, label=d.label, serial=d.serial, measure_resist=single)
            d.link.start()
            d.state, d.message = "starting", f"starting {d.label}…"

    def stop(self) -> None:
        links = [d.link for d in self.links if d.link is not None]
        for d in self.links:
            d.link = None
        for link in links:  # tell them all first, so they shut down in parallel
            link.stop_event.set()
        for link in links:
            link.join(timeout=5.0)  # daemon threads: never hang the app on exit

    def measure_resistance(self) -> bool:
        """Re-measure electrode contact (R). Every streaming device pauses, they're measured one at a
        time, then resume one at a time (~5 s per device; the dead-man stops the rover meanwhile)."""
        ready = [d for d in self.links if d.link is not None and d.state == "streaming"]
        if not ready:
            return False
        rnd = threading.Barrier(len(ready))
        for d in ready:
            d.link.resist_round = rnd
            d.link.resist_request.set()
        return True

    # -- data --------------------------------------------------------------------------
    def _read_chunk(self):
        chunks = [self._drain(d) for d in self.links]
        if self.state != "streaming":
            self.artifact = False
        if self._merger is None:
            ts, xs = chunks[0]
            if not ts:
                return np.zeros(0), np.zeros((0, len(self.sdk_channels)))
            return np.concatenate(ts), np.vstack(xs)
        for k, (ts, xs) in enumerate(chunks):
            if ts:
                self._merger.push(k, np.concatenate(ts), np.vstack(xs))
        return self._merger.pull(self.clock())

    def _drain(self, d: _Device) -> tuple[list, list]:
        ts, xs = [], []
        for _ in range(_MAX_CHUNKS_PER_POLL):
            try:
                msg = d.q.get_nowait()
            except queue.Empty:
                break
            kind = msg[0]
            if kind == "eeg":
                ts.append(msg[1])
                xs.append(msg[2])
            elif kind == "status":
                if msg[1] != d.state or msg[1] == "error":   # console log of connection progress
                    print(f"[neurorover] {d.label}: {msg[1]} - {msg[2]}", flush=True)
                d.state, d.message = msg[1], msg[2]
            elif kind == "device":
                d.info = msg[1]
                d.battery = msg[1].get("battery")
                print(f"[neurorover] {d.label}: {msg[1].get('kind')} {msg[1].get('name')} "
                      f"serial {msg[1].get('serial')}, channels {msg[1].get('channels')}, "
                      f"battery {msg[1].get('battery')}%", flush=True)
            elif kind == "battery":
                d.battery = msg[1]
            elif kind == "resist":
                self._apply_resistance(d, msg[1])
        return ts, xs

    def _apply_resistance(self, d: _Device, by_sdk_name: dict[str, float]) -> None:
        mapping = self.devices[d.name]["map"]
        for n, r in by_sdk_name.items():
            if n in mapping:
                pos = mapping[n]
                self.resistance[pos] = r
                self.contact[pos] = math.isfinite(r) and r < config.EEG_CONTACT_GOOD_OHM

    def poor_contacts(self) -> list[str]:
        return [p for p in self.positions if self.contact.get(p) is False]

    def _battery_text(self) -> str:
        known = [d for d in self.links if d.battery is not None]
        if not known:
            return ""
        if len(self.links) == 1:
            return f", battery {known[0].battery}%"
        return ", battery " + ", ".join(f"{d.label} {d.battery}%" for d in known)

    def status(self) -> SourceStatus:
        if self._fatal:
            return SourceStatus(False, self._fatal)
        streaming = [d for d in self.links if d.state == "streaming"]
        if not streaming:
            if all(d.state == "error" for d in self.links):
                return SourceStatus(False, self.message + " Retrying…")
            return SourceStatus(False, self.message, pending=True)
        bat = self._battery_text()
        warn = self.electrode_warning()
        if warn is not None:
            msg, critical = warn
            # Red (not "pending") while an electrode used for decoding is bad; stays until fixed.
            return SourceStatus(False, f"{msg}{bat}", pending=not critical)
        poor = self.poor_contacts()
        if poor:
            return SourceStatus(False, f"{self._streaming_name(streaming)} streaming, but HIGH RESISTANCE on "
                                       f"{', '.join(poor)} (adjust, then R to re-check){bat}", pending=True)
        missing = [d for d in self.links if d.state != "streaming"]
        if len(self.links) > 1 and self.present_devices:
            # Streaming again, but its data isn't in the decoded windows yet (they span a few seconds).
            missing += [d for d in self.links if d.state == "streaming" and d.name not in self.present_devices]
        if missing:
            others = " · ".join(f"{d.label}: {d.message if d.state != 'streaming' else 'rejoining…'}"
                                for d in missing)
            return SourceStatus(False, f"{self._streaming_name(streaming)} streaming; {others} "
                                       f"(decoding without it){bat}", pending=True)
        return SourceStatus(True, f"{self._streaming_name(streaming)} streaming{bat}")

    def _streaming_name(self, streaming: list[_Device]) -> str:
        if len(self.links) == 1:
            return streaming[0].info.get("kind", "BrainBit")
        return " + ".join(d.label for d in streaming)
