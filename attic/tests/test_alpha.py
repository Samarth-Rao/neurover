"""Alpha mode (eyes closed -> F, eyes open -> S) on the mock EEG."""
import json

import numpy as np
import pytest

import commands as C
import config
from pipeline.alpha import CLOSED, OPEN, AlphaDecoder, alpha_ratio, calibrate_alpha
from pipeline.controller import Controller
from sinks.sim_sink import SimSink
from sources.mock_eeg_source import MockEegSource

FPS = 60


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_alpha_ratio_separates_a_10hz_rhythm_from_broadband_noise():
    fs, n = 250.0, 500
    rng = np.random.default_rng(0)
    t = np.arange(n) / fs
    assert alpha_ratio(20 * np.sin(2 * np.pi * 10 * t) + rng.normal(0, 2, n), fs) > 0.8
    assert alpha_ratio(rng.normal(0, 5, n), fs) < 0.35  # white noise: ~4/26 of the 4-30 Hz power


def test_calibration_keeps_eyes_open_still():
    rng = np.random.default_rng(1)
    cal = calibrate_alpha(rng.normal(0.45, 0.1, 200), rng.normal(0.85, 0.06, 200))
    assert cal.open_recall >= config.ALPHA_CAL_MIN_OPEN_RECALL
    assert 0.55 < cal.threshold < 0.8
    with pytest.raises(ValueError):
        calibrate_alpha([0.5], [0.9])


def _calibrated(tmp_path, seed, user="default"):
    clock = Clock()
    src = MockEegSource(mode="alpha", seed=seed, artifacts=False, clock=clock, models_dir=tmp_path, user=user)
    src.start()
    assert not src.supports_stimulus
    for label, seconds, _ in src.calibration_segments():
        src.on_input(frozenset({"eyes"}) if label == CLOSED else frozenset())
        src.set_calibration_label(label)
        for _ in range(round(seconds * FPS)):
            clock.t += 1.0 / FPS
            src.poll()
    cal = src.finish_calibration()
    return src, clock, cal


@pytest.mark.parametrize("seed", [1, 3, 5])
def test_after_calibration_eyes_closed_drives_and_eyes_open_stops(tmp_path, seed):
    src, clock, cal = _calibrated(tmp_path, seed)
    assert cal.open_recall >= config.ALPHA_CAL_MIN_OPEN_RECALL
    sim = SimSink()
    ctl = Controller(src, [sim], clock=clock)

    def phase(eyes_closed, seconds):
        src.on_input(frozenset({"eyes"}) if eyes_closed else frozenset())
        cmds = []
        for _ in range(round(seconds * FPS)):
            clock.t += 1.0 / FPS
            ctl.tick(clock.t)
            sim.update(1.0 / FPS)
            cmds.append(ctl.command)
        # Measured: up to ~3.5 s to switch (2 s window must turn over, then the 4-window debounce).
        return cmds[round(4.0 * FPS):]

    x0 = sim.sim.x
    for _ in range(2):
        opened = phase(False, 10.0)
        closed = phase(True, 10.0)
        assert np.mean([c == C.STOP for c in opened]) > 0.95
        assert np.mean([c == C.FORWARD for c in closed]) > 0.90
    assert sim.sim.x > x0 + 0.5
    assert phase(False, 6.0)[-1] == C.STOP


def test_alpha_calibration_is_saved_and_reloaded(tmp_path):
    _, _, cal = _calibrated(tmp_path, seed=2, user="ana")
    saved = json.loads(AlphaDecoder.path("ana", tmp_path).read_text())
    assert saved["threshold"] == cal.threshold and saved["user"] == "ana"
    again = MockEegSource(mode="alpha", models_dir=tmp_path, user="ana")
    assert "loaded alpha calibration" in again.load_message
    assert again.decoder.threshold == cal.threshold
    assert again.score_threshold == cal.threshold


def test_alpha_mode_rejects_artifacts_too(tmp_path):
    clock = Clock()
    src = MockEegSource(mode="alpha", seed=1, artifacts=False, clock=clock, models_dir=tmp_path)
    src.start()
    ctl = Controller(src, [SimSink()], clock=clock)
    src.on_input(frozenset({"eyes", "jaw"}))
    cmds = []
    for _ in range(round(6 * FPS)):
        clock.t += 1.0 / FPS
        ctl.tick(clock.t)
        cmds.append(ctl.command)
    assert src.artifact_windows > 0 and src.clean_windows == 0
    assert all(c == C.STOP for c in cmds)


def test_alpha_ui_has_no_flicker_and_calibrates_without_the_warning(tmp_path):
    import pygame

    from ui.app import App

    src = MockEegSource(mode="alpha", seed=1, models_dir=tmp_path)
    sim = SimSink()
    ctl = Controller(src, [sim])
    app = App(src, ctl, [sim], sim=sim)
    assert app.stimulus is None
    seen = {}
    done = set()

    def held(t):
        for at, key in ((0.1, pygame.K_v), (0.3, pygame.K_c)):
            if t >= at and at not in done:
                done.add(at)
                pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key, mod=0))
        if 0.5 < t and "cal" not in seen:
            seen["cal"] = (app.warning_open, app.calibration is not None, ctl.paused)
        return set()

    src.start()
    app.run(duration=0.8, held_provider=held)
    assert seen["cal"] == (False, True, True)   # C: straight into calibration, rover paused
    assert "C calibrate" in app.key_hints() and "V flicker" not in app.key_hints()
