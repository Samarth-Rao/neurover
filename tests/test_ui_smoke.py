"""Headless (SDL dummy driver) run of the real app loop: keyboard -> smoother -> sim + dashboard."""
import commands as C
import config
from pipeline.controller import Controller
from pipeline.session_log import SessionLog
from sinks.sim_sink import SimSink
from sources.keyboard_source import KeyboardSource
from ui.app import App


def test_keyboard_drives_sim_through_full_app_loop(tmp_path):
    source = KeyboardSource()
    sim = SimSink()
    log = SessionLog(tmp_path)
    controller = Controller(source, [sim], logger=log)
    app = App(source, controller, [sim], sim=sim)
    x0 = sim.sim.x

    # Hold "up" for the first second, then release.
    source.start()
    shot = tmp_path / "frame.png"
    app.run(duration=1.5, screenshot=str(shot), held_provider=lambda t: {"up"} if t < 1.0 else set())
    controller.shutdown()

    assert sim.sim.x > x0 + 0.1            # it drove forward (start pose faces +x)
    assert controller.last_sent == C.STOP   # released -> S, plus shutdown S
    assert controller.sent_count >= 12      # ~10 Hz
    assert shot.exists() and shot.stat().st_size > 10_000
    assert log.rows == controller.sent_count
    assert config.WINDOW_SIZE == (1600, 900)


def test_training_keys_record_fit_and_pause(tmp_path):
    """T toggles training (rover paused), held number keys label windows, Enter fits."""
    import pygame

    from sources.mock_emg_source import MockEmgSource

    source = MockEmgSource(seed=3, models_dir=tmp_path)
    sim = SimSink()
    controller = Controller(source, [sim])
    app = App(source, controller, [sim], sim=sim)
    posted = set()
    seen = {}

    def press(t, at, key):
        if t >= at and at not in posted:
            posted.add(at)
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key, mod=0))

    def held(t):
        press(t, 0.05, pygame.K_t)          # training on
        press(t, 2.0, pygame.K_RETURN)      # fit
        if 1.9 < t < 2.0:
            seen["paused"] = controller.paused
            seen["sent"] = controller.last_sent
        if 0.15 <= t < 1.0:
            return {"0"}                    # label rest (virtual arm resting)
        if 1.0 <= t < 1.9:
            return {"1", "up"}              # label fist while the virtual arm makes a fist
        return set()

    source.start()
    app.run(duration=2.3, held_provider=held)
    source.stop()

    counts = source.trainer.data.counts()
    assert counts["rest"] > 5 and counts["fist"] > 5
    assert counts["spread"] == counts["wave_in"] == counts["wave_out"] == 0
    assert seen == {"paused": True, "sent": C.STOP}
    assert source.trainer.clf.is_fitted
    assert app.training and controller.paused


def test_p_toggles_chase_view(tmp_path):
    import pygame

    source = KeyboardSource()
    sim = SimSink()
    controller = Controller(source, [sim])
    app = App(source, controller, [sim], sim=sim)
    assert app.view == "top"
    seen = {}
    posted = set()

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
    assert seen["mid"] == "chase" and app.view == "top"
    assert app.chase_view.camera is not None  # it really drew frames
