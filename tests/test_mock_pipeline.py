"""End to end on synthetic EMG: mock arm -> windows -> features -> KNN -> smoother -> rover."""
import numpy as np

import commands as C
import config
from pipeline.controller import Controller
from sinks.sim_sink import SimSink
from sources.mock_emg_source import MockArm, MockEmgSource

ARROW_FOR = {g: d for d, g in config.MOCK_ARROW_GESTURES.items()}
FPS = 60


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def make(tmp_path, seed=1):
    clock = Clock()
    src = MockEmgSource(seed=seed, clock=clock, models_dir=tmp_path, autoload=False)
    src.start()
    return src, clock


def hold(src, clock, gesture, seconds, label=None, on_frame=None):
    """Hold a virtual gesture (and optionally a training label) for ``seconds`` of fake time."""
    src.on_input(frozenset({ARROW_FOR[gesture]}) if gesture in ARROW_FOR else frozenset())
    src.set_recording(label)
    preds = []
    for _ in range(round(seconds * FPS)):
        clock.t += 1.0 / FPS
        if on_frame:
            on_frame(clock.t)
        else:
            preds += src.poll()
    src.set_recording(None)
    return preds


def train(src, clock, seconds=8.0):
    for g in config.GESTURES:
        hold(src, clock, g, seconds, label=g)
    return src.trainer.fit()


def test_mock_patterns_cover_every_gesture_with_eight_channels():
    assert set(config.MOCK_PATTERNS) == set(config.GESTURES)
    assert all(len(p) == config.EMG_CHANNELS for p in config.MOCK_PATTERNS.values())
    x = MockArm(seed=0).step(0.02)
    assert x.shape == (8,) and np.all(x >= 0)


def test_mock_streams_at_50hz_like_myo_preprocessed(tmp_path):
    src, clock = make(tmp_path)
    for _ in range(4):
        clock.t += 0.5
        assert len(src._read_samples()) == 25


def test_training_ignores_the_settle_period(tmp_path):
    src, clock = make(tmp_path)
    hold(src, clock, "fist", 1.0, label="fist")
    n = src.trainer.data.counts()["fist"]
    expected = (1.0 - config.TRAIN_SETTLE_S) / config.HOP_S
    assert expected - 2 <= n <= expected + 1


def test_untrained_source_predicts_nothing_and_rover_stays_stopped(tmp_path):
    src, clock = make(tmp_path)
    preds = hold(src, clock, "fist", 1.0)
    assert preds and all(p.label is None for p in preds)
    assert preds[-1].emg.shape == (8,) and preds[-1].emg.max() > 100  # signal still shown


def test_end_to_end_accuracy_above_90_percent_on_unseen_data(tmp_path):
    src, clock = make(tmp_path, seed=1)
    report = train(src, clock)
    assert report.cv_accuracy > 0.9

    ok = total = 0
    for seed in (11, 12, 13):
        src.arm = MockArm(seed)  # a fresh "session" the model has never seen
        for g in ("rest", "fist", "wave_in", "spread", "wave_out", "rest", "wave_out", "fist", "spread", "wave_in"):
            preds = hold(src, clock, g, 3.0)
            t0 = preds[0].timestamp
            for p in preds:
                if p.timestamp - t0 >= 0.4:  # skip the transition into the gesture
                    total += 1
                    ok += p.label == g
    accuracy = ok / total
    assert accuracy > 0.90, f"accuracy {accuracy:.3f}"


def test_trained_gestures_drive_the_simulated_rover(tmp_path):
    src, clock = make(tmp_path)
    train(src, clock, seconds=6.0)
    sim = SimSink()
    ctl = Controller(src, [sim], clock=clock)
    x0, h0 = sim.sim.x, sim.sim.heading_deg

    def frame(t):
        ctl.tick(t)
        sim.update(1.0 / FPS)

    hold(src, clock, "rest", 1.0, on_frame=frame)
    assert ctl.command == C.STOP and sim.sim.x == x0
    hold(src, clock, "fist", 2.0, on_frame=frame)
    assert ctl.command == C.FORWARD and sim.sim.x > x0 + 0.3
    hold(src, clock, "wave_in", 1.5, on_frame=frame)   # ~0.4 s gesture->command latency, then ~90 deg/s
    assert ctl.command == C.LEFT and sim.sim.heading_deg > h0 + 45
    hold(src, clock, "rest", 1.0, on_frame=frame)
    assert ctl.command == C.STOP


def test_training_mode_pauses_rover_and_needs_fresh_streak_to_resume(tmp_path):
    src, clock = make(tmp_path)
    train(src, clock, seconds=4.0)
    sim = SimSink()
    ctl = Controller(src, [sim], clock=clock)

    def frame(t):
        ctl.tick(t)
        sim.update(1.0 / FPS)

    hold(src, clock, "fist", 1.0, on_frame=frame)
    assert ctl.command == C.FORWARD
    ctl.set_paused(True)
    hold(src, clock, "fist", 1.0, label="fist", on_frame=frame)
    assert ctl.command == C.STOP and ctl.last_sent == C.STOP
    ctl.set_paused(False)
    ctl.tick(clock.t)
    assert ctl.command == C.STOP
    hold(src, clock, "fist", 0.5, on_frame=frame)
    assert ctl.command == C.FORWARD


# -- guided gesture calibration (C) ------------------------------------------------------------------
def run_guided_calibration(src, clock, cancel_after=None):
    src.start_recording()
    for i, (label, seconds, _) in enumerate(src.calibration_segments()):
        if cancel_after is not None and i >= cancel_after:
            src.cancel_calibration()
            return None
        src.set_calibration_label(label)
        gesture = label if label in ARROW_FOR else None
        src.on_input(frozenset({ARROW_FOR[gesture]}) if gesture else frozenset())
        for _ in range(round(seconds * FPS)):
            clock.t += 1.0 / FPS
            src.poll()
    src.stop_recording()
    return src.apply_calibration(src.fit_calibration())


def test_guided_gesture_calibration_learns_every_gesture_and_sets_the_bars(tmp_path):
    src, clock = make(tmp_path, seed=3)
    segs = src.calibration_segments()
    holds = [s for s in segs if s[0] in src.classes]
    assert len(holds) == config.EMG_CAL_REPS * len(src.classes)
    assert all(a[0] != b[0] for a, b in zip(holds, holds[1:]))           # never the same gesture twice running
    cal = run_guided_calibration(src, clock)
    assert cal.command_accuracy > 0.85 and "right command" in cal.describe()
    assert (tmp_path / "mock_model.joblib").exists()
    # Bars: relaxed ~0, a held gesture fills its strongest channel.
    rest = src.display_levels(cal.relaxed)
    assert rest.max() < 0.05
    preds = hold(src, clock, "fist", 2.0)
    lv = src.display_levels(preds[-1].emg)
    assert lv.max() > 0.5
    # It drives: a held gesture is recognised.
    assert sum(p.label == "fist" for p in preds[-10:]) >= 6        # noisy mock: a clear majority
    # Saved with the model: a new session gets the same bar ranges.
    again = MockEmgSource(seed=4, clock=clock, models_dir=tmp_path)
    assert again.display_cal is not None and np.allclose(again.display_cal[0], cal.relaxed, atol=1e-3)


def test_cancelled_gesture_calibration_keeps_the_previous_model(tmp_path):
    src, clock = make(tmp_path, seed=5)
    train(src, clock, seconds=4.0)
    before = len(src.trainer.data)
    model = src.trainer.clf.model
    assert run_guided_calibration(src, clock, cancel_after=6) is None
    assert len(src.trainer.data) == before and src.trainer.clf.model is model


def test_myo_classes_are_directions_the_user_programs():
    from sources.myo_source import MyoSource

    assert set(MyoSource.command_map.values()) == set(C.ALL)
    assert MyoSource.command_map["relax"] == C.STOP                    # a relaxed arm never drives
    assert set(MyoSource.label_keys.values()) == set(MyoSource.classes)
