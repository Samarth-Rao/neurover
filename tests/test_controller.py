import csv
from typing import Optional

import commands as C
from rover.controller import Controller
from rover.session_log import HEADER, SessionLog
from rover.sim import Arena, RoverSim, SimSink, Sink
from rover.source import Prediction, Source


class ScriptedSource(Source):
    """Emits a new prediction every ``hop`` seconds of fake time, from a script function."""

    name = "scripted"
    classes = C.ALL

    def __init__(self, script, hop=0.05):
        self.script = script
        self.hop = hop
        self.now = 0.0
        self.alive = True
        self._last: Optional[Prediction] = None

    def start(self):
        pass

    def stop(self):
        pass

    def poll(self):
        if self.alive and (self._last is None or self.now - self._last.timestamp >= self.hop):
            label, conf = self.script(self.now)
            self._last = Prediction(label, conf, self.now)
            return [self._last]
        return []


class RecordingSink(Sink):
    name = "rec"

    def __init__(self):
        self.sent = []

    def send(self, command):
        self.sent.append(command)


class BrokenSink(Sink):
    name = "broken"

    def send(self, command):
        raise OSError("port vanished")


def run(ctl: Controller, src: ScriptedSource, seconds: float, fps: float = 60.0, t0: float = 0.0):
    t = t0
    for _ in range(round(seconds * fps)):
        src.now = t
        ctl.tick(t)
        t += 1.0 / fps
    return t


def test_sends_at_fixed_rate_even_when_unchanged():
    src = ScriptedSource(lambda t: (C.STOP, 1.0))
    sink = RecordingSink()
    ctl = Controller(src, [sink], send_hz=10.0)
    run(ctl, src, 3.0)
    assert 29 <= len(sink.sent) <= 31
    assert set(sink.sent) == {C.STOP}


def test_command_reaches_sinks_only_after_debounce():
    src = ScriptedSource(lambda t: (C.FORWARD, 1.0) if t >= 1.0 else (C.STOP, 1.0))
    sink = RecordingSink()
    ctl = Controller(src, [sink])
    run(ctl, src, 1.0)
    assert ctl.command == C.STOP
    t = run(ctl, src, 0.12, t0=1.0)   # ~2-3 windows of F: not yet
    assert ctl.command == C.STOP
    run(ctl, src, 0.5, t0=t)
    assert ctl.command == C.FORWARD
    assert sink.sent[-1] == C.FORWARD


def test_deadman_sends_stop_when_source_goes_silent():
    src = ScriptedSource(lambda t: (C.LEFT, 1.0))
    sink = RecordingSink()
    ctl = Controller(src, [sink])
    t = run(ctl, src, 1.0)
    assert ctl.command == C.LEFT
    src.alive = False
    t = run(ctl, src, 0.4, t0=t)
    assert ctl.command == C.LEFT
    run(ctl, src, 0.3, t0=t)
    assert ctl.command == C.STOP and ctl.smoother.deadman
    assert sink.sent[-1] == C.STOP


def test_broken_sink_does_not_stop_the_others():
    src = ScriptedSource(lambda t: (C.STOP, 1.0))
    good = RecordingSink()
    ctl = Controller(src, [BrokenSink(), good])
    run(ctl, src, 1.0)
    assert len(good.sent) >= 9


def test_session_log_records_every_send_with_blocked_flag(tmp_path):
    # Rover parked 10 cm from a box: F is refused and logged as blocked.
    sim = SimSink(RoverSim(Arena(boxes=[(2.17, 1.0, 2.5, 2.0)]), start=(2.0, 1.5, 0.0)))
    src = ScriptedSource(lambda t: (C.FORWARD, 0.9))
    log = SessionLog(tmp_path)
    ctl = Controller(src, [sim], logger=log)
    run(ctl, src, 2.0)
    ctl.shutdown()
    rows = list(csv.reader(open(log.path, newline="")))
    assert tuple(rows[0]) == HEADER
    body = rows[1:]
    assert len(body) == ctl.sent_count
    assert all(r[1] == "F" and r[2] == "0.900" for r in body)
    assert body[0][3] == "S" and body[-1][3] == "F"          # debounced, then F
    assert body[-1][4] == "1"                                  # ...which the rover refused
    assert all(r[4] == "0" for r in body if r[3] != "F")


def test_shutdown_sends_final_stop():
    src = ScriptedSource(lambda t: (C.FORWARD, 1.0))
    sink = RecordingSink()
    ctl = Controller(src, [sink])
    run(ctl, src, 1.0)
    assert sink.sent[-1] == C.FORWARD
    ctl.shutdown()
    assert sink.sent[-1] == C.STOP
