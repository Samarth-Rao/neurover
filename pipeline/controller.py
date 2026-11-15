"""Ties a source to the sinks: source -> smoother -> fixed-rate command stream.

The UI calls ``tick()`` once per frame. Fresh predictions (newer timestamp)
are fed to the smoother; the smoothed command is sent to every sink at a fixed
SEND_HZ, resent even when unchanged, and every send is logged.
"""
from __future__ import annotations

import sys
import time
from collections import deque
from typing import Callable, Iterable, Optional

import commands
import config
from sinks.base import Sink
from sources.base import Prediction, Source

from .session_log import SessionLog
from .smoother import Smoother


class Controller:
    def __init__(self, source: Source, sinks: Iterable[Sink], *,
                 smoother: Optional[Smoother] = None,
                 logger: Optional[SessionLog] = None,
                 send_hz: float = config.SEND_HZ,
                 history_s: float = config.TIMELINE_S + 1.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.source = source
        self.sinks = list(sinks)
        threshold = getattr(source, "confidence_threshold", None)
        self.smoother = smoother or Smoother(
            threshold=config.CONFIDENCE_THRESHOLD if threshold is None else threshold)
        self.logger = logger
        self.period = 1.0 / send_hz
        self.history_s = history_s
        self.clock = clock

        self.raw: Optional[Prediction] = None        # latest fresh prediction
        self.raw_command: Optional[str] = None       # ...mapped to a command
        self.command: str = commands.STOP            # smoothed command (what we send)
        self.last_sent: Optional[str] = None
        self.sent_count = 0
        self.blocked = False
        # (t, command or None) at each fresh prediction / (t, command, blocked) at each send
        self.raw_history: deque[tuple[float, Optional[str]]] = deque()
        self.sent_history: deque[tuple[float, str, bool]] = deque()

        self.paused = False  # e.g. training mode: keep the pipeline running but send S
        self._last_pred_ts: Optional[float] = None
        self._next_send: Optional[float] = None
        self._sink_errors: set[str] = set()

    def tick(self, now: Optional[float] = None) -> str:
        now = self.clock() if now is None else now
        n = getattr(self.source, "smooth_windows", None)   # e.g. the debounce an EEG calibration chose
        if n and n != self.smoother.n_windows:
            self.smoother.n_windows = int(n)
        for pred in self.source.poll():
            if self._last_pred_ts is not None and pred.timestamp <= self._last_pred_ts:
                continue
            self._last_pred_ts = pred.timestamp
            self.raw = pred
            self.raw_command = commands.from_label(pred.label, self.source.command_map)
            self.smoother.update(self.raw_command, pred.confidence, pred.timestamp)
            self.raw_history.append((now, self.raw_command))

        smoothed = self.smoother.output(now)
        self.command = commands.STOP if self.paused else smoothed

        if self._next_send is None or now >= self._next_send:
            self._send(now)
            # Fixed cadence; if we fell behind (UI stall) don't burst, just resync.
            self._next_send = (now if self._next_send is None else self._next_send) + self.period
            if self._next_send <= now:
                self._next_send = now + self.period

        self._trim(now)
        return self.command

    def set_paused(self, paused: bool) -> None:
        if self.paused and not paused:
            self.smoother.reset()  # resuming needs a fresh, full streak before the rover moves
        self.paused = paused

    def _send(self, now: float) -> None:
        cmd = self.command
        for sink in self.sinks:
            try:
                sink.send(cmd)
            except Exception as exc:  # a broken sink must never stop the others
                if sink.name not in self._sink_errors:
                    self._sink_errors.add(sink.name)
                    print(f"[neurorover] sink {sink.name!r} failed: {exc!r}", file=sys.stderr)
        self.blocked = any(s.blocked for s in self.sinks)
        self.last_sent = cmd
        self.sent_count += 1
        self.sent_history.append((now, cmd, self.blocked))
        if self.logger is not None:
            raw = self.raw
            self.logger.log(raw.label if raw else None, raw.confidence if raw else 0.0, cmd, self.blocked)

    def _trim(self, now: float) -> None:
        cutoff = now - self.history_s
        # Keep one entry older than the cutoff so the timeline's left edge stays filled.
        for hist in (self.raw_history, self.sent_history):
            while len(hist) > 1 and hist[1][0] < cutoff:
                hist.popleft()

    def shutdown(self) -> None:
        """Send a final S to every sink and close the log."""
        for sink in self.sinks:
            try:
                sink.send(commands.STOP)
            except Exception:
                pass
        self.last_sent = commands.STOP
        if self.logger is not None:
            self.logger.close()
