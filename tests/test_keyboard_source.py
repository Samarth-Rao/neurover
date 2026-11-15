import commands as C
from sources.keyboard_source import KeyboardSource


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def make():
    clock = FakeClock()
    src = KeyboardSource(period_s=0.05, clock=clock)
    src.start()
    return src, clock


def test_nothing_held_is_stop():
    src, _ = make()
    src.on_input(frozenset())
    p = src.latest()
    assert p.label == C.STOP and p.confidence == 1.0 and p.emg is None


def test_each_direction_maps_to_its_command():
    for held, cmd in (("up", "F"), ("down", "B"), ("left", "L"), ("right", "R")):
        src, _ = make()
        src.on_input(frozenset({held}))
        assert src.latest().label == cmd


def test_most_recently_pressed_key_wins_and_release_falls_back():
    src, _ = make()
    src.on_input(frozenset({"up"}))
    src.on_input(frozenset({"up", "left"}))
    assert src.current_command() == C.LEFT
    src.on_input(frozenset({"up"}))
    assert src.current_command() == C.FORWARD
    src.on_input(frozenset())
    assert src.current_command() == C.STOP


def test_emits_one_fresh_prediction_per_period():
    src, clock = make()
    a = src.latest()
    clock.t += 0.02
    assert src.latest() is a
    clock.t += 0.04
    b = src.latest()
    assert b is not a and b.timestamp > a.timestamp


def test_stopped_source_produces_no_fresh_data():
    src, clock = make()
    a = src.latest()
    src.stop()
    clock.t += 1.0
    assert src.latest() is a
