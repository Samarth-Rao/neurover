"""Myo source against a fake pyomyo, in a real spawned worker process (no hardware)."""
import time
from pathlib import Path

import numpy as np
import pytest

import commands as C
import config
from pipeline.controller import Controller
from sinks.sim_sink import SimSink
from sources.myo_source import MyoSource

FAKE = str(Path(__file__).parent / "fake_pyomyo")


@pytest.fixture
def myo(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(FAKE)
    made = []

    def make(scenario):
        monkeypatch.setenv("FAKE_MYO_SCENARIO", scenario)
        src = MyoSource(models_dir=tmp_path, log_dir=tmp_path, autoload=False, backend="dongle")
        made.append(src)
        return src

    yield make
    for src in made:
        src.stop()


def pump(src, seconds, ctl=None, until=None):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if ctl is not None:
            ctl.tick()
        else:
            src.poll()
        if until is not None and until():
            return True
        time.sleep(0.01)
    return until is None


def test_streams_preprocessed_emg_through_the_pipeline(myo):
    src = myo("stream")
    src.start()
    assert pump(src, 15, until=lambda: src.windows > 10), src.status()
    st = src.status()
    assert st.ok and "COM9" in st.message and "battery 87%" in st.message
    pred = src.latest()
    assert pred.label is None                       # no model trained yet
    assert np.allclose(pred.emg, [100, 200, 300, 400, 500, 600, 700, 800])
    assert time.monotonic() - pred.timestamp < 1.0  # child timestamps are in our clock


def test_missing_dongle_is_an_on_screen_error_not_a_crash(myo):
    src = myo("no_dongle")
    src.start()
    assert pump(src, 15, until=lambda: src.state == "error")
    st = src.status()
    assert not st.ok and "dongle not found" in st.message.lower()
    assert "Retrying" in st.message


def test_busy_port_is_reported(myo):
    src = myo("busy")
    src.start()
    assert pump(src, 15, until=lambda: src.state == "error")
    assert "Myo Connect" in src.status().message


def test_missing_pyomyo_is_reported_without_spawning(myo, monkeypatch):
    import sources.myo_source as ms

    monkeypatch.setattr(ms.importlib.util, "find_spec", lambda name: None)
    src = myo("stream")
    src.start()
    assert src._proc is None
    st = src.status()
    assert not st.ok and "pip install pyomyo" in st.message


def test_armband_dropout_forces_stop_then_reconnects(myo, monkeypatch):
    monkeypatch.setattr(config, "MYO_RECONNECT_AFTER_S", 1.5)
    src = myo("drop")
    sim = SimSink()
    ctl = Controller(src, [sim])
    src.start()
    assert pump(src, 15, ctl, until=lambda: src.state == "connected" and src.windows > 5)
    # Stream stops after ~1 s: dead-man forces S, then the worker is restarted.
    assert pump(src, 6, ctl, until=lambda: ctl.smoother.deadman)
    assert ctl.command == C.STOP and ctl.last_sent == C.STOP
    assert pump(src, 15, ctl, until=lambda: src.restarts >= 1 and src.state == "connected")


def test_stop_terminates_a_blocked_worker(myo):
    src = myo("drop")
    src.start()
    assert pump(src, 15, until=lambda: src.windows > 5)
    pump(src, 1.5)  # now blocked inside run()
    proc = src._proc
    t0 = time.monotonic()
    src.stop()
    assert not proc.is_alive()
    assert time.monotonic() - t0 < 3.0


# -- the built-in Bluetooth backend (sources/myo_ble_worker.py) ----------------------------------------
def test_ble_notification_parsing_and_timestamps():
    from sources import myo_ble_worker as w

    data = bytes(range(-8, 8)) if False else bytes([(v + 256) % 256 for v in range(-8, 8)])
    a, b = w.parse_emg(data)
    assert a == tuple(range(-8, 0)) and b == tuple(range(0, 8))     # int8, two samples of 8 channels
    assert w.parse_emg(b"\x00" * 10) == []
    clock = w.SampleClock()
    ts = []
    now = 100.0
    for _ in range(400):                       # 2 samples per notification, bursty arrival
        now += 0.01 + (0.004 if len(ts) % 7 == 0 else 0.0)
        ts += clock.stamp(2, now)
    d = np.diff(ts)
    assert np.all(d > 0) and abs(np.median(d) - 1 / 200) < 1e-4  # steady 200 Hz, never backwards
    assert abs(ts[-1] - now) < 0.25                               # tracks arrival time


def test_backend_choice_and_separate_models(monkeypatch, tmp_path):
    import sources.myo_source as ms

    monkeypatch.setattr(ms, "dongle_present", lambda: False)
    assert ms.pick_backend("auto", None) == "ble"
    assert ms.pick_backend("auto", "COM5") == "dongle"
    monkeypatch.setattr(ms, "dongle_present", lambda: True)
    assert ms.pick_backend("auto", None) == "dongle"
    with pytest.raises(ValueError):
        ms.pick_backend("wifi", None)
    ble = MyoSource(backend="ble", models_dir=tmp_path, log_dir=tmp_path, autoload=False)
    dongle = MyoSource(backend="dongle", models_dir=tmp_path, log_dir=tmp_path, autoload=False)
    assert ble.trainer.name == "myo-ble" and dongle.trainer.name == "myo"   # differently scaled EMG: own models
    assert ble.min_full_scale < dongle.min_full_scale


def test_is_myo_matches_by_service_or_name():
    from types import SimpleNamespace as NS
    from sources import myo_ble_worker as w

    dev = NS(name=None, address="AA:BB")
    assert w.is_myo(dev, NS(local_name=None, service_uuids=[w.CONTROL_SERVICE]))
    assert w.is_myo(dev, NS(local_name="My Myo", service_uuids=[]))
    assert not w.is_myo(dev, NS(local_name="BrainBit", service_uuids=[]))
    assert w.is_myo(NS(name="Lab", address="AA:BB"), NS(local_name="Lab", service_uuids=[]), name_hint="lab")
