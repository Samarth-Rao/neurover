import commands as C
from pipeline.smoother import Smoother

HOP = 0.05


def feed(sm: Smoother, seq, t0=0.0, conf=1.0):
    t = t0
    for cmd in seq:
        t += HOP
        sm.update(cmd, conf, t)
    return t


def test_starts_stopped_with_deadman_active():
    sm = Smoother()
    assert sm.output(0.0) == C.STOP
    assert sm.deadman


def test_switches_only_after_n_consecutive_windows():
    sm = Smoother(n_windows=4)
    t = feed(sm, [C.FORWARD] * 3)
    assert sm.output(t) == C.STOP
    t = feed(sm, [C.FORWARD], t)
    assert sm.output(t) == C.FORWARD


def test_interrupted_streak_restarts():
    sm = Smoother(n_windows=4)
    t = feed(sm, [C.FORWARD, C.FORWARD, C.FORWARD, C.LEFT, C.FORWARD, C.FORWARD, C.FORWARD])
    assert sm.output(t) == C.STOP
    t = feed(sm, [C.FORWARD], t)
    assert sm.output(t) == C.FORWARD


def test_flicker_does_not_change_command():
    sm = Smoother(n_windows=4)
    t = feed(sm, [C.FORWARD] * 4)
    t = feed(sm, [C.LEFT, C.RIGHT, C.LEFT, C.BACK, C.LEFT, C.RIGHT], t)
    assert sm.output(t) == C.FORWARD


def test_low_confidence_holds_previous_command():
    sm = Smoother(n_windows=4, threshold=0.6)
    t = feed(sm, [C.FORWARD] * 4)
    t = feed(sm, [C.LEFT] * 10, t, conf=0.59)
    assert sm.output(t) == C.FORWARD


def test_low_confidence_window_breaks_the_streak():
    sm = Smoother(n_windows=4, threshold=0.6)
    t = feed(sm, [C.LEFT] * 3)
    t = feed(sm, [C.LEFT], t, conf=0.2)
    t = feed(sm, [C.LEFT] * 3, t)
    assert sm.output(t) == C.STOP
    t = feed(sm, [C.LEFT], t)
    assert sm.output(t) == C.LEFT


def test_confidence_exactly_at_threshold_counts():
    sm = Smoother(n_windows=2, threshold=0.6)
    t = feed(sm, [C.RIGHT] * 2, conf=0.6)
    assert sm.output(t) == C.RIGHT


def test_no_opinion_holds_previous_command():
    sm = Smoother(n_windows=2)
    t = feed(sm, [C.BACK] * 2)
    t = feed(sm, [None] * 5, t)
    assert sm.output(t) == C.BACK


def test_deadman_forces_stop_when_data_stops():
    sm = Smoother(n_windows=4, deadman_s=0.5)
    t = feed(sm, [C.FORWARD] * 4)
    assert sm.output(t + 0.49) == C.FORWARD
    assert not sm.deadman
    assert sm.output(t + 0.51) == C.STOP
    assert sm.deadman


def test_after_deadman_a_full_fresh_streak_is_required():
    sm = Smoother(n_windows=4, deadman_s=0.5)
    t = feed(sm, [C.FORWARD] * 4)
    t += 1.0
    assert sm.output(t) == C.STOP
    t = feed(sm, [C.FORWARD] * 3, t)
    assert sm.output(t) == C.STOP
    assert not sm.deadman
    t = feed(sm, [C.FORWARD], t)
    assert sm.output(t) == C.FORWARD
