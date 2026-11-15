"""The rover app: keyboard + EEG sources through the controller, the window (headless), the flicker
stimulus and its photosensitivity gate, and a guided calibration end to end. No hardware needed."""
import math

import numpy as np
import pygame
import pytest

import commands as C
import config
from eeg import ssvep
from eeg.devices import Feeds
from rover.controller import Controller
from rover.session_log import SessionLog
from rover.sim import SimSink
from rover.source import EegSource, KeyboardSource
from ui.app import App
from ui.stimulus import Stimulus, StimulusLayout, frame_pattern

POSTERIOR = {"flex": {**config.EEG_DEVICES["flex"], "map": {"O1": "PO3", "O2": "PO4", "T3": "POz", "T4": "Pz"}},
             "headband": config.EEG_DEVICES["headband"]}


class Clock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


def eeg_source(tmp_path, preset="clean", clock=None, seed=1, **kw):
    clock = clock or Clock()
    feeds = Feeds(("flex", "headband"), POSTERIOR, mock=True, mock_preset=preset, clock=clock, seed=seed)
    return EegSource(feeds, models_dir=tmp_path, recordings_dir=tmp_path, clock=clock, **kw), clock


# -- keyboard ----------------------------------------------------------------------------------------
def test_keyboard_maps_keys_and_the_newest_wins():
    clock = Clock()
    src = KeyboardSource(period_s=0.05, clock=clock)
    src.start()
    assert src.poll()[0].label == C.STOP
    for held, cmd in (("up", "F"), ("down", "B"), ("left", "L"), ("right", "R")):
        src.on_input(frozenset({held}))
        assert src.current_command() == cmd
    src.on_input(frozenset({"up"}))
    src.on_input(frozenset({"up", "left"}))
    assert src.current_command() == C.LEFT
    src.on_input(frozenset())
    assert src.current_command() == C.STOP
    clock.t += 0.02
    assert src.poll() == []                         # one prediction per period
    clock.t += 0.04
    assert len(src.poll()) == 1
    src.stop()
    clock.t += 1.0
    assert src.poll() == []                         # stopped: no fresh data -> the dead-man stops the rover


# -- EEG source through the controller ----------------------------------------------------------------
def test_looking_at_a_box_drives_and_the_centre_stops(tmp_path):
    src, clock = eeg_source(tmp_path)
    sim = SimSink()
    ctl = Controller(src, [sim], clock=clock)
    src.start()
    seen = {}
    for k in range(150):                            # 15 s: centre, then the ← box, then the centre
        clock.t += 0.1
        gaze = "left" if 50 <= k < 100 else None
        src.on_input(frozenset({gaze}) if gaze else frozenset())
        ctl.tick(clock.t)
        if k in (45, 95, 145):
            seen[k] = ctl.command
    src.stop()
    assert seen == {45: C.STOP, 95: C.LEFT, 145: C.STOP}
    assert set(src.decoding) == {"flex", "headband"}
    assert src.status().ok


def test_a_silent_device_is_left_out_and_nothing_at_all_trips_the_dead_man(tmp_path):
    src, clock = eeg_source(tmp_path)
    ctl = Controller(src, [], clock=clock)
    src.start()
    for _ in range(40):
        clock.t += 0.1
        ctl.tick(clock.t)
    hb = src.feeds.device("headband")
    real = src.feeds._mock.pump

    def without_headband(now):                      # the headband's Bluetooth drops
        real(now)
        while not hb.q.empty():
            hb.q.get_nowait()
    src.feeds._mock.pump = without_headband
    for _ in range(30):
        clock.t += 0.1
        ctl.tick(clock.t)
    assert set(src.decoding) == {"flex"} and not ctl.smoother.deadman
    st = src.status()
    assert not st.ok and st.pending and "decoding without it" in st.message
    src.feeds._mock.pump = lambda now: None          # and now the Flex too (silent after 1 s, dead-man 0.5 s later)
    for _ in range(25):
        clock.t += 0.1
        ctl.tick(clock.t)
    assert ctl.smoother.deadman and ctl.command == C.STOP


def test_uncalibrated_mock_at_rest_never_drives(tmp_path):
    src, clock = eeg_source(tmp_path, preset="realistic", seed=4)
    ctl = Controller(src, [], clock=clock)
    src.start()
    sent = set()
    for _ in range(600):                            # a minute of looking at the centre
        clock.t += 0.1
        ctl.tick(clock.t)
        sent.add(ctl.command)
    assert sent == {C.STOP}


# -- calibration end to end ------------------------------------------------------------------------------
def _short_calibration(monkeypatch):
    monkeypatch.setattr(config, "SSVEP_CAL_PER_CLASS", 4)
    monkeypatch.setattr(config, "SSVEP_CAL_PRACTICE", 1)
    monkeypatch.setattr(config, "SSVEP_CAL_LEAD_S", 1.0)
    monkeypatch.setattr(config, "SSVEP_CAL_BREAK_EVERY", 0)


def test_guided_calibration_records_fits_saves_and_reloads(tmp_path, monkeypatch):
    from eeg import cues

    _short_calibration(monkeypatch)
    src, clock = eeg_source(tmp_path)
    src.start()
    run = cues.Run(src.calibration_protocol(), clock.t, seed=1)
    src.start_calibration()
    while not run.finished:
        clock.t += 0.05
        run.update(clock.t)
        st = run.step
        src.on_input(frozenset({st.label}) if st.kind == "cue" and st.label in src.targets else frozenset())
        src.poll()
    rec, path = src.finish_calibration({**run.meta(), "refresh_hz": 60.0})
    assert path.exists() and rec["meta"]["kind"] == "ssvep-calibration"
    assert len(rec["meta"]["trials"]) == src.calibration_protocol().n_trials + 1
    assert rec["meta"]["protocol"]["task"] == "look"
    rec["path"] = path
    s = src.fit_calibration(rec)
    assert s.report["chosen"]["hit"] >= 0.75 and s.setup == src.setup() and s.report["calibrations"] == 1
    src.apply_calibration(s)
    again, _ = eeg_source(tmp_path)
    assert again.settings.calibrated and again.settings.threshold == s.threshold
    assert "loaded" in again.load_message
    # calibrating again adds to the first one (same electrodes); another placement's isn't mixed in
    other = EegSource(Feeds(("flex",), {"flex": {**POSTERIOR["flex"], "map": {"O1": "O1", "O2": "O2", "T3": "Oz",
                                                                             "T4": "POz"}}},
                            mock=True, clock=clock, seed=2), models_dir=tmp_path, recordings_dir=tmp_path, clock=clock)
    assert other.earlier_calibrations() == []
    assert again.earlier_calibrations(exclude=path) == []
    second = __import__("eeg.recording", fromlist=["x"]).save(rec, tmp_path, "ssvep-calibration-default")
    assert again.earlier_calibrations(exclude=second) == [path]
    s2 = again.fit_calibration({**__import__("eeg.recording", fromlist=["x"]).load(second), "path": second})
    assert s2.report["calibrations"] == 2 and s2.report["trials"] == 2 * s.report["trials"]
    assert "from 2 calibrations" in s2.describe()


def test_window_runs_the_cued_calibration(tmp_path, monkeypatch):
    _short_calibration(monkeypatch)
    monkeypatch.setattr(config, "SSVEP_CAL_AWAY_S", 0.3)
    monkeypatch.setattr(config, "SSVEP_CAL_AWAY_JITTER_S", 0.0)
    monkeypatch.setattr(config, "SSVEP_CAL_BASELINE_S", 0.3)
    monkeypatch.setattr(config, "SSVEP_CAL_CUE_S", 0.4)
    src, _ = eeg_source(tmp_path)
    src.clock = src.feeds.clock = src.feeds._mock.clock = __import__("time").monotonic
    sim = SimSink()
    app = App(src, Controller(src, [sim]), [sim], sim=sim)
    app.stimulus.warning_accepted = True
    seen, done = {}, set()

    def held(t):
        for at, key in ((0.2, pygame.K_c), (0.5, pygame.K_SPACE)):      # C starts, Space skips the get-ready
            if t >= at and at not in done:
                done.add(at)
                pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key, mod=0))
        run = app.calibration
        if run is not None and not run.finished:
            seen.setdefault(run.step.kind, t)
            for kind in ("fixation", "cue"):          # the 3-2-1 with its arrow, and the framed box
                if run.step.kind == kind and f"shot {kind}" not in seen:
                    seen[f"shot {kind}"] = True
                    app.draw()
                    app.save_screenshot(tmp_path / f"{kind}.png")
        st = run.step if run is not None else None
        return {st.label} if st is not None and st.kind == "cue" and st.label in src.targets else set()

    src.start()
    app.run(duration=2.0, held_provider=held)
    src.stop()
    assert seen["ready"] < 0.3 and seen["away"] < 0.7                   # Space skipped the 1 s get-ready
    assert seen["away"] < seen["fixation"] < seen["cue"]               # the rover, the 3-2-1 + arrow, the box
    assert "pause" not in seen and (tmp_path / "fixation.png").exists() and (tmp_path / "cue.png").exists()
    assert app.controller.paused                                       # the rover is stopped while calibrating


# -- the window, headless ------------------------------------------------------------------------------
def test_keyboard_drives_the_sim_through_the_full_app_loop(tmp_path):
    source = KeyboardSource()
    sim = SimSink()
    log = SessionLog(tmp_path)
    controller = Controller(source, [sim], logger=log)
    app = App(source, controller, [sim], sim=sim)
    x0 = sim.sim.x
    source.start()
    shot = tmp_path / "frame.png"
    app.run(duration=1.5, screenshot=str(shot), held_provider=lambda t: {"up"} if t < 1.0 else set())
    controller.shutdown()
    assert sim.sim.x > x0 + 0.1                     # drove forward (the start pose faces +x)
    assert controller.last_sent == C.STOP
    assert controller.sent_count >= 12               # ~10 Hz
    assert shot.exists() and shot.stat().st_size > 10_000
    assert log.rows == controller.sent_count


def test_p_toggles_the_chase_view():
    source = KeyboardSource()
    sim = SimSink()
    app = App(source, Controller(source, [sim]), [sim], sim=sim)
    seen, posted = {}, set()

    def held(t):
        for at in (0.2, 0.8):
            if t >= at and at not in posted:
                posted.add(at)
                pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_p, mod=0))
        if 0.5 < t < 0.7:
            seen["mid"] = app.view
        return {"up"} if t < 0.6 else set()

    source.start()
    app.run(duration=1.0, held_provider=held)
    assert seen["mid"] == "chase" and app.view == "top" and app.chase_view.camera is not None


def test_mock_eeg_window_with_flicker_decoder_view_and_warning_gate(tmp_path):
    src, _ = eeg_source(tmp_path)
    src.clock = __import__("time").monotonic
    src.feeds.clock = src.clock
    src.feeds._mock.clock = src.clock
    sim = SimSink()
    ctl = Controller(src, [sim])
    app = App(src, ctl, [sim], sim=sim)
    plan = [(0.1, pygame.K_v), (0.3, pygame.K_ESCAPE), (0.5, pygame.K_v), (0.7, pygame.K_RETURN), (0.9, pygame.K_i)]
    checks = {0.2: "after_v", 0.4: "after_escape", 0.8: "after_enter"}
    done, seen = set(), {}

    def held(t):
        for at, key in plan:
            if t >= at and at not in done:
                done.add(at)
                pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key, mod=0))
        for at, name in checks.items():
            if t >= at and name not in seen:
                seen[name] = (app.warning_open, app.stimulus.enabled, app.running)
        return {"left"} if t > 1.0 else set()

    src.start()
    shot = tmp_path / "eeg.png"
    app.run(duration=2.5, screenshot=str(shot), held_provider=held)
    src.stop()
    assert seen["after_v"] == (True, False, True)          # V shows the warning, no flicker yet
    assert seen["after_escape"] == (False, False, True)    # Esc cancels
    assert seen["after_enter"] == (False, True, True)      # Enter accepts: flicker on
    assert app.show_decoding and shot.stat().st_size > 10_000


# -- stimulus ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("frames,hz", [(7, 8.57), (6, 10.0), (5, 12.0), (4, 15.0)])
def test_stimulus_frame_patterns_have_the_right_cycle_length(frames, hz):
    pat = frame_pattern(frames)
    assert len(pat) == frames and sum(pat) == (frames + 1) // 2
    stim = Stimulus({"x": frames})
    seq = []
    for _ in range(frames * 10):
        seq.append(stim.is_on("x"))
        stim.advance()
    assert seq == pat * 10
    assert stim.frequencies(60.0)["x"] == pytest.approx(hz, abs=0.01)


def test_stimulus_layout_boxes_are_big_and_far_apart():
    lay = StimulusLayout.fit(pygame.Rect(0, 0, 1600, 900), 220, 16)
    boxes = list(lay.boxes.values())
    assert all(b.w >= 200 and b.h >= 200 for b in boxes)
    c = {k: v.center for k, v in lay.boxes.items()}
    assert min(math.dist(c[a], c[b]) for a in ("up", "down") for b in ("left", "right")) > 700
    others = [lay.centre, *lay.panels.values()]
    assert not any(b.colliderect(o) for b in boxes for o in others)


# -- live camera (main.py --camera) ----------------------------------------------------------------
class FakeCapture:
    """Stands in for cv2.VideoCapture: a moving gradient, BGR, 640x480 at ~60 fps."""

    def __init__(self, index):
        self.index, self.n, self.released = index, 0, False

    def read(self):
        __import__("time").sleep(1 / 60)
        self.n += 1
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, 2] = (np.arange(640) + self.n) % 256          # red ramp (BGR)
        return True, frame

    def release(self):
        self.released = True


def _bgr_to_rgb(frame):
    return np.ascontiguousarray(frame[:, :, ::-1])


def _fake_opener(available=(0, 1)):
    opened = []

    def opener(index):
        if index not in available:
            return None, f"camera {index} isn't available"
        cap = FakeCapture(index)
        opened.append(cap)
        return cap, None
    return opener, opened


def test_camera_reads_frames_in_the_background_and_switches():
    import time

    from rover.camera import Camera

    opener, opened = _fake_opener()
    cam = Camera(0, opener=opener, prepare=_bgr_to_rgb)
    cam.start()
    deadline = time.monotonic() + 3
    while cam.latest()[0] is None and time.monotonic() < deadline:
        time.sleep(0.02)
    frame, _ = cam.latest()
    assert frame.shape == (480, 640, 3) and frame[:, :, 0].max() > 200 and frame[:, :, 2].max() == 0   # now RGB
    time.sleep(0.3)
    live, text = cam.status()
    assert live and "camera 0" in text and cam.fps > 10
    cam.next()                                                   # K: camera 1
    assert cam.index == 1 and opened[0].released and len(opened) == 2
    cam.next()                                                   # camera 2 doesn't exist: say so, don't crash
    live, text = cam.status()
    assert not live and "camera 2" in text
    cam.stop()


def test_camera_view_in_the_window_and_p_cycles_views(tmp_path):
    import time

    from rover.camera import Camera

    opener, _ = _fake_opener()
    cam = Camera(0, opener=opener, prepare=_bgr_to_rgb)
    source = KeyboardSource()
    sim = SimSink()
    app = App(source, Controller(source, [sim]), [sim], sim=sim, view="camera", camera=cam)
    seen, done = [], set()

    def held(t):
        for at in (0.6, 0.9, 1.2):
            if t >= at and at not in done:
                done.add(at)
                seen.append(app.view)
                pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_p, mod=0))
        if 0.5 < t < 0.55 and "shot" not in done:
            done.add("shot")
            app.draw()
            app.save_screenshot(tmp_path / "camera.png")
        return set()

    source.start()
    app.run(duration=1.5, held_provider=held)
    assert seen == ["camera", "top", "chase"] and app.view == "camera"
    assert (tmp_path / "camera.png").stat().st_size > 10_000 and cam._thread is None   # stopped on exit


def test_camera_calibrations_are_kept_apart_from_sim_ones(tmp_path):
    sim_src, clock = eeg_source(tmp_path)
    cam_src = EegSource(Feeds(("flex", "headband"), POSTERIOR, mock=True, clock=clock, seed=1),
                        models_dir=tmp_path, recordings_dir=tmp_path, view="camera", clock=clock)
    assert cam_src.setup() == sim_src.setup() + " | view:camera"
    sim_src.start()
    sim_src.start_calibration()
    for _ in range(40):
        clock.t += 0.05
        sim_src.poll()
    rec, path = sim_src.finish_calibration({"trials": [], "protocol": sim_src.calibration_protocol().to_meta()})
    assert rec["meta"]["view"] == "sim"
    assert sim_src.earlier_calibrations() == [path] and cam_src.earlier_calibrations() == []


def test_clenching_the_jaw_stops_the_rover_at_once_while_looking_at_a_box(tmp_path):
    src, clock = eeg_source(tmp_path)
    ctl = Controller(src, [], clock=clock)
    src.start()
    seen = {}
    for k in range(160):                            # look at ↑ throughout; clench from 8 to 10 s
        clock.t += 0.1
        src.on_input(frozenset({"up", "jaw"}) if 80 <= k < 100 else frozenset({"up"}))
        ctl.tick(clock.t)
        seen[k] = (ctl.command, src.stopped)
    src.stop()
    assert seen[79] == (C.FORWARD, None)
    assert seen[83] == (C.STOP, "muscle")            # within 0.4 s, no debounce
    assert all(seen[k][0] == C.STOP for k in range(83, 100))
    assert seen[159] == (C.FORWARD, None)           # driving again after the hold + a fresh streak


def test_calibrations_with_other_flicker_frequencies_are_not_combined(tmp_path):
    from eeg import recording

    src, clock = eeg_source(tmp_path)
    src.start()
    src.start_calibration()
    for _ in range(40):
        clock.t += 0.05
        src.poll()
    rec, path = src.finish_calibration({"trials": [], "protocol": src.calibration_protocol().to_meta()})
    assert src.earlier_calibrations() == [path]
    rec["meta"]["frames"] = {**config.SSVEP_FRAMES, "right": 4}           # recorded when → was 15 Hz
    recording.save(rec, tmp_path, "ssvep-calibration-default", replace=path)
    assert src.earlier_calibrations() == []
