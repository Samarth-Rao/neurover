"""Thalmic Myo armband source (8-channel EMG).

Two ways to reach the armband (config.MYO_BACKEND, --myo-backend):
    dongle  the Bluegiga USB adapter, via pyomyo, PREPROCESSED mode (50 Hz)  sources/myo_worker.py
    ble     this computer's own Bluetooth, via bleak, filtered EMG (200 Hz)  sources/myo_ble_worker.py
    auto    the dongle if one is plugged in, else Bluetooth
Either way the radio runs in its own process (on Windows the dongle's serial port may only be used
from one thread) and EMG samples come back through a multiprocessing.Queue. Hardware problems never
crash the app: they show up in ``status()`` (on screen), the dead-man stops the rover, and the
worker is restarted to try again. The two backends' EMG is scaled differently, so each keeps its
own trained model (models "myo" and "myo-ble").
"""
from __future__ import annotations

import importlib.util
import multiprocessing as mp
import queue
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import config

from . import myo_ble_worker, myo_worker
from .base import SourceStatus
from .signal_source import SignalSource

_MAX_MESSAGES_PER_POLL = 5000


BACKENDS = ("auto", "dongle", "ble")
BLUEGIGA_USB = (0x2458, 0x0001)   # the Myo dongle's USB vendor/product id


def dongle_present() -> bool:
    try:
        from serial.tools import list_ports
        return any((p.vid, p.pid) == BLUEGIGA_USB for p in list_ports.comports())
    except Exception:
        return False


def pick_backend(backend: str, port: Optional[str]) -> str:
    if backend not in BACKENDS:
        raise ValueError(f"unknown Myo backend {backend!r}; choose from {', '.join(BACKENDS)}")
    if backend != "auto":
        return backend
    return "dongle" if port or dongle_present() else "ble"


class MyoSource(SignalSource):
    name = "myo"
    # The person chooses a gesture per command (guided calibration: C); see config.MYO_CLASSES.
    classes = config.MYO_CLASSES
    command_map = config.MYO_CLASS_COMMANDS
    class_names = config.MYO_CLASS_NAMES
    label_keys = config.MYO_LABEL_KEYS

    def __init__(self, port: Optional[str] = None, *, backend: str = config.MYO_BACKEND,
                 clock: Callable[[], float] = time.monotonic, worker: Optional[Callable] = None,
                 log_dir=config.LOGS_DIR, **kwargs) -> None:
        self.backend = pick_backend(backend, port)
        super().__init__(config.EMG_CHANNELS, "myo-ble" if self.backend == "ble" else "myo", clock=clock, **kwargs)
        if self.backend == "ble":
            self.min_full_scale = self.signal_full_scale = config.MYO_BLE_MIN_FULL_SCALE
        self.port = port if self.backend == "dongle" else (port or config.MYO_BLE_NAME)
        self._worker = worker or (myo_ble_worker.run if self.backend == "ble" else myo_worker.run)
        self._log_dir = Path(log_dir)
        self._ctx = mp.get_context("spawn")  # same behaviour on Windows, macOS and Linux
        self._proc: Optional[mp.process.BaseProcess] = None
        self._q = None
        self._stop = None
        self._running = False
        self.state = "idle"        # idle | starting | scanning | connected | error
        self.message = ""
        self.dongle: Optional[str] = None
        self.battery: Optional[int] = None
        self.restarts = 0
        self._state_since = clock()
        self._last_sample: Optional[float] = None
        self._clock_offset: Optional[float] = None
        self._fatal = False

    # -- lifecycle ---------------------------------------------------------------------
    def start(self) -> None:
        self._running = True
        lib = "bleak" if self.backend == "ble" else "pyomyo"
        if importlib.util.find_spec(lib) is None:
            self._set("error", f"{lib} is not installed. Run: pip install {lib}")
            self._fatal = True
            return
        self._spawn()

    def stop(self) -> None:
        self._running = False
        self._kill()

    def _spawn(self) -> None:
        self._kill()
        log_path = None
        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            log_path = str(self._log_dir / f"myo_{datetime.now():%Y%m%d}.log")
        except OSError:
            pass
        self._q = self._ctx.Queue()
        self._stop = self._ctx.Event()
        self._proc = self._ctx.Process(target=self._worker, args=(self._q, self._stop, self.port, log_path),
                                       name="myo-worker", daemon=True)
        self._proc.start()
        self._last_sample = None
        self._set("starting", "starting Myo process…")

    def _kill(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None:
            self._stop.set()
            proc.join(timeout=0.5)       # usually blocked inside a serial read: terminate it
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1.0)
        if self._q is not None:
            try:
                self._q.cancel_join_thread()
                self._q.close()
            except Exception:
                pass
            self._q = None

    def _set(self, state: str, message: str) -> None:
        if state != self.state or message != self.message:
            if state != self.state or state == "error":
                print(f"[neurorover] Myo: {state} - {message}", flush=True)
            self.state, self.message = state, message
            self._state_since = self.clock()

    # -- data --------------------------------------------------------------------------
    def _local_time(self, t_child: float) -> float:
        # time.monotonic() is system-wide on Windows/macOS/Linux, so child timestamps are directly
        # comparable. Guard anyway: if the clocks disagree by >1 s, re-base the child's clock.
        now = self.clock()
        if self._clock_offset is None:
            self._clock_offset = (now - t_child) if abs(now - t_child) > 1.0 else 0.0
        return min(t_child + self._clock_offset, now)

    def _read_samples(self):
        out = []
        if self._q is not None:
            for _ in range(_MAX_MESSAGES_PER_POLL):
                try:
                    msg = self._q.get_nowait()
                except (queue.Empty, OSError, EOFError, ValueError):
                    break
                kind = msg[0]
                if kind == "emg":
                    out.append((self._local_time(msg[1]), msg[2]))
                elif kind == "scanning":
                    self.dongle = msg[1]
                    where = "over Bluetooth" if msg[1] == "Bluetooth" else f"(dongle on {msg[1]})"
                    self._set("scanning", f"scanning {where} for the armband…")
                elif kind == "connected":
                    self.dongle = msg[1]
                    self._set("connected", f"connected via {msg[1]}")
                elif kind == "battery":
                    self.battery = msg[1]
                elif kind == "error":
                    self._set("error", msg[1])
        now = self.clock()
        if out:
            self._last_sample = now
        self._supervise(now)
        return out

    def _supervise(self, now: float) -> None:
        """Restart the worker after errors, or when a connected armband stops sending EMG."""
        if not self._running or self._fatal:
            return
        if self._proc is not None and not self._proc.is_alive() and self.state != "error":
            self._set("error", "Myo process exited unexpectedly (see logs/myo_*.log).")
        waited = now - self._state_since
        stalled = (self.state == "connected"
                   and now - (self._last_sample or self._state_since) > config.MYO_RECONNECT_AFTER_S)
        if (self.state == "error" and waited > config.MYO_RETRY_S) or stalled:
            self.restarts += 1
            self._spawn()

    def status(self) -> SourceStatus:
        now = self.clock()
        if self.state == "error":
            retry = "" if self._fatal else f" Retrying every {config.MYO_RETRY_S:g} s."
            return SourceStatus(False, self.message + retry)
        if self.state in ("idle", "starting"):
            return SourceStatus(False, "starting Myo process…", pending=True)
        if self.state == "scanning":
            msg = self.message
            if now - self._state_since > config.MYO_SCAN_WARN_S:
                msg = "still scanning: is the armband on, charged and nearby? Tap the logo to wake it."
                if self.backend == "ble":
                    msg += " Asleep? Plug its USB cable in for a moment, then unplug it."
            return SourceStatus(False, msg, pending=True)
        since = now - (self._last_sample or self._state_since)
        if since > config.MYO_NO_DATA_WARN_S:
            return SourceStatus(False, f"connected but no EMG for {since:.0f} s (out of range?)")
        bat = f", battery {self.battery}%" if self.battery is not None else ""
        return SourceStatus(True, f"{self.message}{bat}")
