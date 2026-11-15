"""BrainBit SDK thread: scan, connect, measure contact, stream, reconnect (one per device).

pyneurosdk2 API used here (confirmed against the installed 1.0.15 package source,
sdk.brainbit.com and the official samples at gitlab.com/neurosdk2/neurosamples):

    from neurosdk.scanner import Scanner
    from neurosdk.cmn_types import SensorFamily, SensorCommand, SensorState, SensorGain, ...
    scanner = Scanner([SensorFamily.LEBrainBit, SensorFamily.LEBrainBitFlex])
    scanner.start(); infos = scanner.sensors(); scanner.stop()     # SensorInfo(SensFamily, SensModel, Name, ...)
    sensor = scanner.create_sensor(info)       # blocking, connects
    # Flex 4/8 -> LEBrainBitFlex -> BrainBit2Sensor:
    #   sensor.supported_channels -> [EEGChannelInfo(Id, ChType, Name, Num)]
    #   sensor.signalDataReceived = fn(sensor, [SignalChannelsData(PackNum, Marker, Samples[Num] in volts)])
    #   sensor.resistDataReceived = fn(sensor, [ResistRefChannelsData(PackNum, Samples, Referents)])
    #   sensor.amplifier_param = BrainBit2AmplifierParam(ChSignalMode, ChResistUse, ChGain, Current)
    # Flex v1 -> LEBrainBit (firmware major > 100) -> BrainBitSensor:
    #   signalDataReceived = fn(sensor, [BrainBitSignalData(PackNum, Marker, O1, O2, T3, T4)])  # volts
    #   resistDataReceived = fn(sensor, BrainBitResistData(O1, O2, T3, T4))                      # one object
    sensor.exec_command(SensorCommand.StartSignal / StopSignal / StartResist / StopResist)
    sensor.sensorStateChanged = fn(sensor, SensorState.StateInRange | StateOutOfRange)
    sensor.connect() / sensor.disconnect()     # the SDK does NOT reconnect by itself

Rules followed: SDK calls only from this thread (create_sensor/connect/exec_command
block); callbacks only copy data into queues; signal and resistance never run at
the same time; sensor is deleted before the scanner.

Two devices at once (a Flex and the headband): one BrainBitLink thread each. A Flex v1 and the
headband are the SAME SDK family (LEBrainBit, both named "BrainBit"); only the firmware tells them
apart (major > 100 = Flex v1), and that is known only after connecting. So: a link can be pinned to
a serial number (config.EEG_DEVICES[...]["serial"]); devices are "claimed" so two links never
connect to the same one; and a link that connects to the other kind hands it over (disconnects
and remembers what it is) and keeps looking. Scanner creation, connecting and disconnecting are
serialised across the threads with one lock, in case the native library minds doing two at once.

Messages put on ``out`` (tuples):
    ("status", state, text)      state: scanning | connecting | resist | streaming | reconnecting | error
    ("device", dict)             kind, name, serial, channels (SDK names in our order), battery
    ("resist", {name: ohms})
    ("battery", percent)
    ("eeg", t[n], microvolts[n, channels])
"""
from __future__ import annotations

import math
import queue
import sys
import threading
import time
from typing import Optional, Sequence

import numpy as np

import config

_SDK_LOCK = threading.RLock()
FAMILIES = ("flex", "headband", "any")
_CLAIM_LOCK = threading.Lock()
# One device at a time goes from "connecting" to "delivering samples". Measured on the real pair
# (Flex v1 + headband, macOS): a contact (resistance) measurement overlapping the other device
# connecting or measuring leaves the Flex silent until it's power-cycled. Streaming side by side is fine.
_BRINGUP = threading.Lock()
# Contact checks (R) with two devices: every device stops streaming, they're measured ONE AT A TIME,
# then they restart one at a time. Verified on the real pair: both stream normally afterwards.
_RESIST_LOCK = threading.Lock()
_CLAIMS: dict[str, str] = {}       # device address -> link name connected / connecting to it
_KIND_OF: dict[str, str] = {}      # device address -> "flex" | "headband", once seen connected


class _WrongDevice(Exception):
    """Connected to the other kind of BrainBit; hand it over and keep looking."""


class BrainBitLink(threading.Thread):
    def __init__(self, out: "queue.Queue", channel_names: Sequence[str], device: Optional[str] = None,
                 fs: float = config.EEG_SAMPLE_HZ, family: str = "any", label: str = "BrainBit",
                 serial: Optional[str] = None, measure_resist: bool = True) -> None:
        super().__init__(name=f"brainbit-{label.lower()}", daemon=True)
        if family not in FAMILIES:
            raise ValueError(f"unknown BrainBit family {family!r}; choose from {', '.join(FAMILIES)}")
        self.out = out
        self.channel_names = list(channel_names)
        self.device = device
        self.fs = fs
        self.family = family
        self.label = label
        self.serial = str(serial) if serial else None
        self.measure_resist = measure_resist   # at startup; off with two devices (see _BRINGUP)
        self.resist_round: Optional[threading.Barrier] = None   # shared by all devices for one R press
        self._claimed: Optional[str] = None
        self._bringup = False
        self._samples_seen = 0
        self.stop_event = threading.Event()
        self.resist_request = threading.Event()
        self._state_q: "queue.Queue" = queue.Queue()
        self._resist_rows: list[list[float]] = []
        self._next_t: Optional[float] = None
        self._last_sample = 0.0

    # -- helpers ---------------------------------------------------------------------
    def put(self, *msg) -> None:
        self.out.put(msg)

    def status(self, state: str, text: str) -> None:
        self.put("status", state, text)

    def _sleep(self, seconds: float) -> bool:
        """Sleep unless asked to stop. Returns False if stopping."""
        return not self.stop_event.wait(seconds)

    def _push_samples(self, rows: list[list[float]]) -> None:
        """Called on the SDK's callback thread: timestamp and enqueue, nothing else."""
        n = len(rows)
        if n == 0:
            return
        now = time.monotonic()
        # Sample clock at the device rate. BLE delivers in bursts, and the device crystal and our
        # clock differ slightly, so slew gently towards arrival time (no jumps = no broken windows);
        # only hard-reset after a real gap (> 0.5 s, e.g. lost packets), so the gap stays visible.
        if self._next_t is None:
            self._next_t = now - (n - 1) / self.fs
        error = now - (self._next_t + (n - 1) / self.fs)
        if abs(error) > 0.5:
            self._next_t += error
        elif error > 0:
            self._next_t += 0.05 * error  # only ever move forwards, so timestamps stay increasing
        t = self._next_t + np.arange(n) / self.fs
        self._next_t = float(t[-1]) + 1.0 / self.fs
        self._last_sample = now
        self._samples_seen += n
        self.put("eeg", t, np.asarray(rows, dtype=float) * 1e6)  # volts -> microvolts

    # -- thread ----------------------------------------------------------------------
    def run(self) -> None:
        try:
            from neurosdk.cmn_types import SensorFamily  # noqa: F401
            from neurosdk.scanner import Scanner  # noqa: F401
        except Exception as exc:  # ImportError, or the native library failing to load
            self.status("error", f"BrainBit SDK not available ({exc}). Run: pip install pyneurosdk2")
            return
        while not self.stop_event.is_set():
            try:
                self._session()
            except _WrongDevice as exc:
                self.status("scanning", f"{exc}; looking for the {self.label}…")
                if not self._sleep(1.0):
                    return
            except Exception as exc:
                msg = f"BrainBit: {exc}"
                if "Code 108" in str(exc):  # seen on macOS right after a previous connection; a retry works
                    msg += " (often transient; if it persists on macOS, put the headband in pairing mode)"
                self.status("error", msg)
                if not self._sleep(config.BRAINBIT_RECONNECT_S):
                    return

    def _families(self):
        from neurosdk.cmn_types import SensorFamily

        # A Flex v1 reports the headband's family, so the Flex slot looks at both.
        return {"flex": [SensorFamily.LEBrainBitFlex, SensorFamily.LEBrainBit],
                "headband": [SensorFamily.LEBrainBit],
                "any": [SensorFamily.LEBrainBit, SensorFamily.LEBrainBitFlex]}[self.family]

    def _wanted(self, info) -> bool:
        addr = str(info.Address)
        if self.serial and str(info.SerialNumber) != self.serial:
            return False
        if self.device and self.device.lower() not in f"{info.Name} {info.SerialNumber}".lower():
            return False
        if _CLAIMS.get(addr, self.name) != self.name:
            return False                     # another link has it
        kind = _KIND_OF.get(addr)
        return not (kind and self.family != "any" and kind != self.family)

    def _claim(self, info) -> bool:
        addr = str(info.Address)
        with _CLAIM_LOCK:
            if _CLAIMS.get(addr, self.name) != self.name:
                return False
            _CLAIMS[addr] = self.name
            self._claimed = addr
            return True

    def _release(self) -> None:
        with _CLAIM_LOCK:
            if self._claimed and _CLAIMS.get(self._claimed) == self.name:
                del _CLAIMS[self._claimed]
            self._claimed = None

    def _end_bringup(self) -> None:
        if self._bringup:
            self._bringup = False
            _BRINGUP.release()

    def _wait_first_samples(self, timeout: float = 3.0) -> None:
        """Hold the bring-up lock until samples actually flow, then let the other device start."""
        seen, deadline = self._samples_seen, time.monotonic() + timeout
        while self._samples_seen == seen and time.monotonic() < deadline and not self.stop_event.is_set():
            time.sleep(0.05)
        self._end_bringup()

    @staticmethod
    def _kind(sensor) -> str:
        from neurosdk.cmn_types import SensorFamily

        if sensor.sens_family == SensorFamily.LEBrainBitFlex:
            return "flex"
        try:
            return "flex" if sensor.version.FwMajor > 100 else "headband"
        except Exception:
            return "headband"

    def _session(self) -> None:
        from neurosdk.scanner import Scanner

        with _SDK_LOCK:
            scanner = Scanner(self._families())
        sensor = None
        try:
            info = self._scan(scanner)
            if info is None:
                return
            self.status("connecting", f"connecting to {info.Name} ({info.SerialNumber or info.Address})…")
            while not _BRINGUP.acquire(timeout=0.5):   # the other device is still starting up
                if self.stop_event.is_set():
                    return
            self._bringup = True
            with _SDK_LOCK:
                sensor = scanner.create_sensor(info)  # blocking; connects
            kind = self._kind(sensor)
            _KIND_OF[str(info.Address)] = kind
            if self.family != "any" and kind != self.family:
                raise _WrongDevice(f"{info.Name} {info.SerialNumber} is the {kind}")
            self._run_sensor(sensor, info)
        finally:
            self._end_bringup()
            if sensor is not None:
                self._cleanup(sensor)
                del sensor
            self._release()
            with _SDK_LOCK:
                del scanner

    def _scan(self, scanner):
        with _SDK_LOCK:
            scanner.start()
        try:
            t0 = time.monotonic()
            while not self.stop_event.is_set():
                infos = sorted((i for i in scanner.sensors() if self._wanted(i)), key=lambda i: -i.RSSI)
                for info in infos:
                    if self._claim(info):
                        return info
                waited = time.monotonic() - t0
                if waited > config.BRAINBIT_SCAN_WARN_S:
                    hint = f"Is the {self.label} switched on and nearby?"
                    if self.serial:
                        hint += f" (looking for serial {self.serial})"
                    if sys.platform == "darwin":
                        hint += " macOS: allow Bluetooth for your terminal app (Settings > Privacy & Security > Bluetooth)."
                    self.status("scanning", f"still scanning ({waited:.0f} s). {hint}")
                else:
                    self.status("scanning", f"scanning for the {self.label}…")
                self._sleep(0.5)
            return None
        finally:
            with _SDK_LOCK:
                scanner.stop()

    def _run_sensor(self, sensor, info) -> None:
        from neurosdk.cmn_types import SensorCommand, SensorFamily, SensorState

        family = sensor.sens_family
        if family == SensorFamily.LEBrainBitFlex:
            extract, resist_extract, kind, sdk_names = self._setup_flex4(sensor, info)
        else:
            extract, resist_extract, kind, sdk_names = self._setup_brainbit(sensor)

        def on_signal(_sensor, data):
            try:
                self._push_samples([extract(d) for d in data])
            except Exception:
                pass

        def on_resist(_sensor, data):
            try:
                for d in (data if isinstance(data, list) else [data]):
                    self._resist_rows.append(resist_extract(d))
            except Exception:
                pass

        def on_state(_sensor, state):
            self._state_q.put(state)

        def on_battery(_sensor, level):
            self.put("battery", int(level))

        sensor.signalDataReceived = on_signal
        sensor.resistDataReceived = on_resist
        sensor.sensorStateChanged = on_state
        sensor.batteryChanged = on_battery
        battery = None
        try:
            battery = int(sensor.batt_power)
        except Exception:
            pass
        self.put("device", {"kind": kind, "name": info.Name, "serial": info.SerialNumber,
                            "channels": sdk_names, "battery": battery})

        if self.measure_resist:
            self._measure_resistance(sensor)
        self._start_signal(sensor)
        self._wait_first_samples()
        while not self.stop_event.is_set():
            if self.resist_request.is_set():
                self.resist_request.clear()
                self._contact_round(sensor)
            lost = False
            try:
                while True:
                    lost = self._state_q.get_nowait() == SensorState.StateOutOfRange or lost
            except queue.Empty:
                pass
            if lost or time.monotonic() - self._last_sample > config.BRAINBIT_STALL_S:
                self._reconnect(sensor)
            self._sleep(0.1)

    def _setup_flex4(self, sensor, info):
        from neurosdk.cmn_types import (BrainBit2AmplifierParam, BrainBit2ChannelMode, GenCurrent,
                                        SensorGain)

        chans = sensor.supported_channels
        by_name = {c.Name: c.Num for c in chans}
        if all(n in by_name for n in self.channel_names):
            nums = [by_name[n] for n in self.channel_names]
            names = list(self.channel_names)
        elif len(chans) == len(self.channel_names):
            # Names don't match the device's "map" in config.EEG_DEVICES: keep the SDK's order, say so.
            nums = [c.Num for c in chans]
            names = list(self.channel_names)
            self.status("connecting", f"SDK channel names {[c.Name for c in chans]} differ from the map in config.EEG_DEVICES "
                                      f"{self.channel_names}; using the SDK order")
        else:
            raise RuntimeError(f"device reports channels {[c.Name for c in chans]}; "
                               f"set that device's \"map\" in config.EEG_DEVICES to match")
        n = len(chans)
        try:  # Gain 3 for Flex (BrainBit's recommendation), normal mode, resistance measurement enabled.
            try:
                amp = sensor.amplifier_param
            except Exception:  # pyneurosdk2 can't read back a channel in "short" mode
                amp = BrainBit2AmplifierParam(ChSignalMode=[], ChResistUse=[], ChGain=[], Current=GenCurrent.GenCurr6nA)
            amp.ChSignalMode = [BrainBit2ChannelMode.ChModeNormal] * n
            amp.ChResistUse = [True] * n
            amp.ChGain = [SensorGain.Gain3] * n
            amp.Current = GenCurrent.GenCurr6nA
            sensor.amplifier_param = amp
        except Exception as exc:
            self.status("connecting", f"couldn't set amplifier gain ({exc}); using the device default")
        kind = f"BrainBit Flex (model {info.SensModel})"
        return (lambda d: [d.Samples[i] for i in nums]), (lambda d: [d.Samples[i] for i in nums]), kind, names

    def _setup_brainbit(self, sensor):
        from neurosdk.cmn_types import SensorGain

        fixed = ["O1", "O2", "T3", "T4"]
        if sorted(self.channel_names) != sorted(fixed):
            raise RuntimeError(f"this device always reports {fixed}; use those names as the device's \"map\" keys in config.EEG_DEVICES")
        v = sensor.version
        is_flex = v.FwMajor > 100
        kind = "BrainBit Flex (v1)" if is_flex else "BrainBit headband"
        if is_flex:
            try:
                sensor.gain = SensorGain.Gain3
            except Exception:
                pass
        names = list(self.channel_names)
        return ((lambda d: [getattr(d, n) for n in names]), (lambda d: [getattr(d, n) for n in names]),
                kind, names)

    def _contact_round(self, sensor) -> None:
        """R: stop streaming, wait for every device to stop, measure one device at a time, wait for all,
        then restart streaming one device at a time (see _RESIST_LOCK)."""
        from neurosdk.cmn_types import SensorCommand

        rnd = self.resist_round
        sensor.exec_command(SensorCommand.StopSignal)
        self.status("resist", "waiting to measure electrode contact…")

        def sync() -> None:
            if rnd is not None:
                try:
                    rnd.wait(timeout=20.0)
                except threading.BrokenBarrierError:
                    pass

        sync()                                    # everyone has stopped streaming
        with _RESIST_LOCK:
            self._measure_resistance(sensor)
        sync()                                    # everyone has been measured
        while not _BRINGUP.acquire(timeout=0.5):
            if self.stop_event.is_set():
                return
        self._bringup = True
        self._start_signal(sensor)
        self._wait_first_samples()

    def _start_signal(self, sensor) -> None:
        from neurosdk.cmn_types import SensorCommand

        self._next_t = None
        self._last_sample = time.monotonic()
        sensor.exec_command(SensorCommand.StartSignal)
        self.status("streaming", "streaming")

    def _measure_resistance(self, sensor) -> None:
        from neurosdk.cmn_types import SensorCommand

        self.status("resist", "measuring electrode contact…")
        self._resist_rows = []
        sensor.exec_command(SensorCommand.StartResist)
        try:
            self._sleep(config.BRAINBIT_RESIST_S)
        finally:
            sensor.exec_command(SensorCommand.StopResist)
        rows = [r for r in self._resist_rows if len(r) == len(self.channel_names)]
        result = {}
        for i, name in enumerate(self.channel_names):
            vals = [r[i] for r in rows if r[i] is not None and math.isfinite(r[i])]
            result[name] = float(np.median(vals)) if vals else math.inf
        self.put("resist", result)

    def _reconnect(self, sensor) -> None:
        from neurosdk.cmn_types import SensorState

        self.status("reconnecting", "connection lost: reconnecting…")
        while not self.stop_event.is_set():
            try:
                if sensor.state != SensorState.StateInRange:
                    with _SDK_LOCK:
                        sensor.connect()
                self._start_signal(sensor)
                return
            except Exception as exc:
                self.status("reconnecting", f"connection lost: reconnecting… ({exc})")
                self._sleep(config.BRAINBIT_RECONNECT_S)

    def _cleanup(self, sensor) -> None:
        """Stop and disconnect, then wait for the SDK's own disconnect callback BEFORE the sensor object
        is released. pyneurosdk2 delivers that callback asynchronously on a native thread; if the Python
        sensor is already freed, the callback reads freed memory and the process segfaults (seen on macOS)."""
        from neurosdk.cmn_types import SensorCommand, SensorState

        sensor.signalDataReceived = None
        sensor.resistDataReceived = None
        try:
            sensor.exec_command(SensorCommand.StopSignal)
        except Exception:
            pass
        while not self._state_q.empty():
            self._state_q.get_nowait()
        try:
            with _SDK_LOCK:
                sensor.disconnect()
        except Exception:
            pass
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:  # sensorStateChanged (our on_state) is still attached here
            try:
                if self._state_q.get(timeout=0.1) == SensorState.StateOutOfRange:
                    break
            except queue.Empty:
                pass
        time.sleep(0.2)  # let any trailing native callbacks drain
        for attr in ("sensorStateChanged", "batteryChanged"):
            try:
                setattr(sensor, attr, None)
            except Exception:
                pass
