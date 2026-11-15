"""SSVEP decoding on the mock EEG (no hardware)."""
import json

import numpy as np
import pytest

import commands as C
import config
from pipeline import ssvep
from pipeline.controller import Controller
from sinks.sim_sink import SimSink
from sources.mock_eeg_source import MockEegSource

FPS = 60


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def make(tmp_path, seed=1, **kw):
    clock = Clock()
    src = MockEegSource(seed=seed, artifacts=False, clock=clock, models_dir=tmp_path, **kw)
    src.start()
    return src, clock


def look(src, clock, gaze, seconds, ctl=None, held=()):
    """Virtual user stares at ``gaze`` (None = centre). Returns the predictions produced."""
    src.on_input(frozenset(({gaze} if gaze else set()) | set(held)))
    preds = []
    for _ in range(round(seconds * FPS)):
        clock.t += 1.0 / FPS
        if ctl is not None:
            before = len(ctl.raw_history)
            ctl.tick(clock.t)
            preds += [None] * (len(ctl.raw_history) - before)
        else:
            preds += src.poll()
    return preds


def test_cca_finds_a_clean_sinusoid():
    fs, n = 250.0, 500
    t = np.arange(n) / fs
    x = np.column_stack([np.sin(2 * np.pi * 12 * t + 0.3), 0.5 * np.sin(2 * np.pi * 12 * t + 1.0)])
    x += 0.05 * np.random.default_rng(0).normal(size=x.shape)
    assert ssvep.cca_max_corr(x, ssvep.reference_signals(12, n, fs)) > 0.95
    assert ssvep.cca_max_corr(x, ssvep.reference_signals(15, n, fs)) < 0.3


def test_flat_channel_is_ignored_not_fatal():
    fs, n = 250.0, 500
    t = np.arange(n) / fs
    x = np.column_stack([np.sin(2 * np.pi * 10 * t), np.zeros(n)])
    assert ssvep.cca_max_corr(x, ssvep.reference_signals(10, n, fs)) > 0.95
    assert ssvep.cca_max_corr(np.zeros((n, 3)), ssvep.reference_signals(10, n, fs)) == 0.0


def test_target_frequencies_from_frame_counts():
    f = ssvep.target_frequencies(config.SSVEP_FRAMES, 60.0)
    assert f == pytest.approx({"up": 60 / 7, "down": 10.0, "left": 12.0, "right": 15.0})


@pytest.mark.parametrize("gaze", ["up", "down", "left", "right"])
def test_cca_picks_the_correct_target_over_90_percent_at_2s_windows(tmp_path, gaze):
    src, clock = make(tmp_path, seed=3)
    assert src.window_s == 2.0
    preds = look(src, clock, gaze, 30.0)
    settled = [p for p in preds if p.timestamp - 1000.0 > 3.0]  # window filled with the response
    assert len(settled) > 80
    picks = [max(p.probs, key=p.probs.get) for p in settled]
    accuracy = np.mean([pk == gaze for pk in picks])
    assert accuracy > 0.90, f"{gaze}: {accuracy:.2f}"


def test_low_signal_yields_stop(tmp_path):
    src, clock = make(tmp_path, seed=5)
    src.gen.ssvep_uv = 0.15  # the user is looking at a box but the response is buried in noise
    sim = SimSink()
    ctl = Controller(src, [sim], clock=clock)
    sent = []
    src.on_input(frozenset({"up"}))
    for _ in range(round(30.0 * FPS)):
        clock.t += 1.0 / FPS
        ctl.tick(clock.t)
        sim.update(1.0 / FPS)
        sent.append(ctl.command)
    labels = [label for _, label in ctl.raw_history]
    assert np.mean([label == C.STOP for label in labels]) > 0.75   # most windows: threshold + margin say rest
    assert np.mean([c == C.STOP for c in sent]) > 0.95             # and the command sent stays S


def test_strong_gaze_drives_through_the_controller(tmp_path):
    src, clock = make(tmp_path, seed=2)
    sim = SimSink()
    ctl = Controller(src, [sim], clock=clock)
    assert ctl.smoother.threshold == float("-inf")  # EEG decoders gate themselves
    src.gen.ssvep_uv = 4.0                # a clear response: this test is about the plumbing
    x0 = sim.sim.x
    src.on_input(frozenset({"up"}))
    for _ in range(round(8.0 * FPS)):
        clock.t += 1.0 / FPS
        ctl.tick(clock.t)
        sim.update(1.0 / FPS)
    assert ctl.command == C.FORWARD
    assert sim.sim.x > x0 + 0.5


def test_calibrate_prefers_keeping_rest_still():
    rng = np.random.default_rng(0)
    labels = ["up", "down"]

    def scores(hi):
        s = {l: rng.normal(0.35, 0.03) for l in labels}
        if hi:
            s[hi] = rng.normal(0.6, 0.05)
        return s

    samples = {"rest": [scores(None) for _ in range(200)],
               "up": [scores("up") for _ in range(200)],
               "down": [scores("down") for _ in range(200)]}
    cal = ssvep.calibrate(samples, labels)
    assert 0.40 < cal.threshold < 0.60
    assert cal.rest_recall >= config.SSVEP_CAL_MIN_REST_RECALL
    assert cal.balanced_accuracy > 0.9


def test_guided_calibration_sets_saves_and_reloads(tmp_path):
    src, clock = make(tmp_path, seed=7, user="sam")
    for label in ["rest", *src.classes]:
        src.set_calibration_label(label)
        look(src, clock, None if label == "rest" else label, config.SSVEP_CAL_SEGMENT_S)
    counts = {k: len(v) for k, v in src.cal_samples.items()}
    expected = (config.SSVEP_CAL_SEGMENT_S - config.SSVEP_CAL_SKIP_S) / config.SSVEP_HOP_S
    assert all(abs(n - expected) <= 2 for n in counts.values()), counts
    cal = src.finish_calibration()
    assert src.decoder.threshold == cal.threshold and src.decoder.margin == cal.margin
    assert cal.rest_recall >= config.SSVEP_CAL_MIN_REST_RECALL

    path = ssvep.calibration_path("sam", tmp_path)
    saved = json.loads(path.read_text())
    assert saved["user"] == "sam" and saved["window_s"] == 2.0

    again, _ = make(tmp_path, user="sam")
    assert "loaded SSVEP calibration" in again.load_message
    assert again.decoder.threshold == cal.threshold
    other_window, _ = make(tmp_path, user="sam", window_s=3.0)
    assert "ignored" in other_window.load_message
    assert other_window.decoder.threshold == config.SSVEP_THRESHOLD_FUSED   # default (two devices, averaged)


def test_negative_rest_scores_still_stop_the_rover():
    """Regression: calibrated scores (log-likelihood ratios) are negative when nothing is looked at.
    With a 0 confidence gate those rest windows were ignored and the last drive command stuck."""
    from pipeline.smoother import Smoother
    from sources.eeg_common import EegStreamSource

    sm = Smoother(n_windows=3, threshold=EegStreamSource.confidence_threshold)
    for i in range(5):
        sm.update(C.RIGHT, 12.0, 100 + i * 0.1)
    assert sm.output(100.45) == C.RIGHT
    for i in range(3):
        sm.update(C.STOP, -4.0, 100.5 + i * 0.1)
    assert sm.output(100.75) == C.STOP
