"""The shared EEG layer (eeg/): devices (mock and the fake BrainBit SDK), noise handling, SSVEP
scoring + calibration, recordings. No hardware needed."""
import sys
import time
from pathlib import Path

import numpy as np
import pytest

import config
from eeg import devices, recording, ssvep
from eeg.clean import FLAT, NOISY, OK, SATURATED, Cleaner, without_shared
from eeg.filters import StreamFilter

FS = 250.0
FAKE = str(Path(__file__).parent / "fake_neurosdk")
UNPINNED = {n: {k: v for k, v in d.items() if k != "serial"} for n, d in config.EEG_DEVICES.items()}
POSTERIOR = {"flex": {**config.EEG_DEVICES["flex"], "map": {"O1": "PO3", "O2": "PO4", "T3": "POz", "T4": "Pz"}},
             "headband": config.EEG_DEVICES["headband"]}


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


# -- filters ----------------------------------------------------------------------------------
def test_stream_filter_removes_dc_and_mains_and_is_chunk_invariant():
    t = np.arange(int(10 * FS)) / FS
    x = (300 + 10 * np.sin(2 * np.pi * 12 * t) + 40 * np.sin(2 * np.pi * 60 * t))[:, None]
    whole = StreamFilter(FS, config.SSVEP_BAND, config.EEG_NOTCH_HZ).process(x)
    f = StreamFilter(FS, config.SSVEP_BAND, config.EEG_NOTCH_HZ)
    parts = np.vstack([f.process(x[i:i + 37]) for i in range(0, len(x), 37)])
    assert np.allclose(whole, parts)
    tail = whole[int(5 * FS):, 0]
    assert abs(tail.mean()) < 0.5 and 6.0 < tail.std() < 8.0       # 10 µV sine survives (7.07 RMS), DC + 60 Hz gone


# -- noise handling -------------------------------------------------------------------------------
def _feed(cl, make, seconds=8.0, hop=0.1):
    for k in range(int(seconds / hop)):
        cl.update(k * hop, *make(k * hop))


def test_shared_reference_noise_flags_nothing():
    rng = np.random.default_rng(0)
    cl = Cleaner(4)
    _feed(cl, lambda t: (rng.normal(0, 5, (500, 4)) + rng.normal(0, 60, (500, 1)),))   # 60 µV on every channel
    assert cl.status == [OK] * 4
    assert cl.shared_level > 40 and np.all(cl.level < 10)


def test_one_bad_electrode_is_flagged_alone_and_recovers_after_reseating():
    rng = np.random.default_rng(1)
    cl = Cleaner(4)

    def make(t):
        x = rng.normal(0, 5, (500, 4)) + rng.normal(0, 30, (500, 1))
        if 2.0 <= t < 10.0:
            x[:, 2] += rng.normal(0, 30, 500)
        return (x,)

    _feed(cl, make, seconds=9.0)
    assert cl.status == [OK, OK, NOISY, OK]
    assert cl.usable() == [0, 1, 3]
    for k in range(90, 180):
        cl.update(k * 0.1, *make(k * 0.1))
    assert cl.status == [OK] * 4


def test_median_of_the_others_keeps_one_bad_electrode_out_of_the_rest():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 5, (2000, 4))
    x[:, 0] += rng.normal(0, 50, 2000)
    rr = without_shared(x)
    assert rr[:, 0].std() > 40 and np.all(rr[:, 1:].std(axis=0) < 10)   # a plain mean would smear it to ~13 µV


def test_flat_and_saturated_electrodes():
    rng = np.random.default_rng(3)
    cl = Cleaner(4)

    def make(t):
        x = rng.normal(0, 5, (500, 4))
        x[:, 1] = 0.01 * rng.normal(size=500)          # lead disconnected
        raw = x.copy()
        raw[:, 3] = 400_000.0                          # amplifier railed
        return x, raw

    _feed(cl, make)
    assert cl.status[1] == FLAT and cl.status[3] == SATURATED and cl.usable() == [0, 2]


def test_spikes_are_down_weighted_not_dropped():
    rng = np.random.default_rng(4)
    cl = Cleaner(3)
    _feed(cl, lambda t: (rng.normal(0, 5, (500, 3)),), seconds=3.0)
    x = rng.normal(0, 5, (500, 3))
    x[200:210] += 300.0                                # a pop on every channel
    w = cl.weights(x)
    assert np.all(w[200:210] < 0.01) and np.mean(w[:190] == 1.0) > 0.95


# -- SSVEP scoring ---------------------------------------------------------------------------------
def test_control_frequencies_stay_clear_of_the_boxes():
    targets = list(ssvep.target_frequencies().values())
    ctl = ssvep.control_frequencies(targets)
    assert len(ctl) >= 6
    for c in ctl:
        assert all(abs(c - h * f) > 0.45 for f in targets for h in (0.5, 1, 2))


def test_a_flicker_stands_out_on_its_own_box_only():
    rng = np.random.default_rng(5)
    t = np.arange(500) / FS
    sc = ssvep.Scorer(ssvep.target_frequencies())
    f = ssvep.target_frequencies()["left"]
    x = rng.normal(0, 5, (500, 4)) + 3 * np.sin(2 * np.pi * f * t)[:, None] * np.array([1, 0.8, 0.6, 0.2])
    z = sc.device_z(t, x)
    assert np.argmax(z) == sc.labels.index("left") and z.max() > 4
    noise = sc.device_z(t, rng.normal(0, 5, (500, 4)))
    assert noise.max() < 3


def test_pooling_adds_agreeing_devices_and_a_silent_one_only_dilutes():
    a = np.array([0.0, 0.0, 5.0, 0.0])
    assert ssvep.pool([a, a])[2] == pytest.approx(5 * np.sqrt(2))
    assert ssvep.pool([a, np.zeros(4)])[2] == pytest.approx(5 / np.sqrt(2))
    assert ssvep.decide(np.array([0.5, 4.5, 4.2, 0.0]), 4.0, 0.5) == -1       # too close to the runner-up
    assert ssvep.decide(np.array([0.5, 4.5, 3.0, 0.0]), 4.0, 0.5) == 1


def _session(names, seed, *, preset="realistic", device_map=POSTERIOR, seconds=60, extra=None):
    """Drive mock devices through 5 s gaze trials (box / centre) and score like live."""
    clk = Clock()
    f = devices.Feeds(names, device_map, mock=True, mock_preset=preset, clock=clk, seed=seed)
    if extra:
        extra(f)
    f.start()
    f.poll()
    sc = ssvep.Scorer(ssvep.target_frequencies())
    dd = {d.name: ssvep.DeviceDecoder(len(d.sdk), sc) for d in f.devices}
    labels = list(ssvep.target_frequencies())
    rng = np.random.default_rng(seed)
    plan = [rng.choice(labels + [None, None]) for _ in range(seconds // 5)]
    picks, decoded = [], 0
    for k in range(seconds * 10):
        clk.t += 0.1
        gaze = plan[min(len(plan) - 1, k // 50)]
        f.mock_input(gaze=gaze)
        f.poll()
        zs = [z for d in f.devices
              for z in [dd[d.name].step(*d.window(None, 2.0, "decode"), d.window(None, 2.0, "raw")[1], FS, 2.0)]
              if z is not None]
        decoded += bool(zs)
        if zs and gaze is not None and k % 50 >= 25 and k > 30:
            picks.append(labels[int(np.argmax(ssvep.pool(zs)))] == gaze)
    return float(np.mean(picks)), decoded / (seconds * 10), {n: d.cleaner.status for n, d in dd.items()}


def _worse_reference(factor, device="headband"):
    def apply(f):
        head = f._mock.head
        k = [d.name for d in f.devices].index(device)
        orig = head._reference

        def ref(n):
            out = orig(n)
            out[:, k] *= factor
            return out
        head._reference = ref
    return apply


def test_a_second_device_helps_and_a_noisy_one_does_not_wreck_it():
    """The bug this refactor fixes: adding the (noisier) headband made everything worse."""
    seeds = (1, 2, 3, 4, 5, 6)                  # the realistic mock is weak: two seeds can tie by chance
    flex = np.mean([_session(("flex",), s)[0] for s in seeds])
    both = np.mean([_session(("flex", "headband"), s)[0] for s in seeds])
    noisy = [_session(("flex", "headband"), s, extra=_worse_reference(5.0)) for s in seeds]
    assert both > flex
    assert np.mean([r[0] for r in noisy]) > flex - 0.05
    assert all(r[1] > 0.95 for r in noisy)                           # no window thrown away
    assert all(st == [OK] * 4 for r in noisy for st in r[2].values())  # shared noise flags no electrode


def test_a_bad_electrode_sits_out_without_taking_its_device_with_it():
    def bad_o2(f):
        head = f._mock.head
        i = head.positions.index("O2")
        gen, rng = head.generate, np.random.default_rng(9)

        def noisy(n):
            x = gen(n)
            x[:, i] += rng.normal(0, 60, n)
            return x
        head.generate = noisy

    acc, decoded, status = _session(("flex", "headband"), 1, preset="clean", extra=bad_o2)
    assert status["headband"] == [OK, NOISY, OK, OK] and status["flex"] == [OK] * 4
    assert decoded > 0.95 and acc > 0.8


# -- calibration -------------------------------------------------------------------------------------
def _calibration_recording(seed, ssvep_uv=None, per_class=4):
    """A mock calibration in the cued design: the virtual driver follows the cues, recorded like the rover does."""
    import dataclasses

    from eeg import cues

    clk = Clock()
    f = devices.Feeds(("flex", "headband"), POSTERIOR, mock=True, mock_preset="clean", clock=clk, seed=seed)
    if ssvep_uv is not None:
        f._mock.head.ssvep_uv = ssvep_uv
    f.start()
    f.poll()
    f.start_recording()
    proto = dataclasses.replace(ssvep.calibration_protocol(), per_class=per_class, practice=1, lead_in_s=1.0,
                                break_every=0)
    run = cues.Run(proto, clk.t, seed=seed)
    boxes = ssvep.target_frequencies()
    while not run.finished:
        clk.t += 0.05
        run.update(clk.t)
        st = run.step
        f.mock_input(gaze=st.label if st.kind == "cue" and st.label in boxes else None)
        f.poll()
    rec = f.stop_recording()
    rec["meta"].update(run.meta())
    return rec


def test_calibration_trials_rest_at_each_fixation_and_skip_pauses_and_practice():
    meta = {"trials": [{"label": "up", "baseline": 0.0, "cue": 2.0, "end": 7.0, "practice": True},
                       {"label": "left", "baseline": 9.0, "cue": 11.0, "end": 16.0},
                       {"label": "rest", "baseline": 18.0, "cue": 20.0, "end": 25.0}]}
    tr = ssvep.calibration_trials(meta)
    assert [(t.label, t.onset, t.end, t.prev) for t in tr] == [
        ("rest", 9.0, 11.0, "rest"), ("left", 11.0, 16.0, "rest"), ("rest", 18.0, 20.0, "left"), ("rest", 20.0, 25.0, "rest")]
    p = ssvep.calibration_protocol()
    assert set(p.classes) == set(ssvep.target_frequencies()) and p.per_class == config.SSVEP_CAL_PER_CLASS


def test_calibration_finds_a_safe_setting_and_refuses_without_a_response(tmp_path):
    rec = _calibration_recording(1)
    s = ssvep.fit(rec, windows=(1.0, 2.0))
    c = s.report["chosen"]
    assert c["hit"] >= 0.75 and c["false"] <= config.SSVEP_CAL_MAX_FALSE and c["safe"]
    assert s.window_s in (1.0, 2.0) and s.n_smooth in config.SSVEP_CAL_SMOOTH
    # saved like the rover does, it reloads only for the same electrodes
    s.setup = ssvep.setup_key({"flex": ["PO3"], "headband": ["O1"]})
    ssvep.save_settings(s, tmp_path)
    back, msg = ssvep.load_settings("default", s.setup, tmp_path)
    assert back.threshold == s.threshold and "loaded" in msg
    other, msg = ssvep.load_settings("default", "flex:C3", tmp_path)
    assert other is None and "other electrodes" in msg
    with pytest.raises(ValueError, match="no reliable flicker response"):
        ssvep.fit(_calibration_recording(2, ssvep_uv=0.0), windows=(2.0,))


def test_replay_matches_a_saved_recording(tmp_path):
    rec = _calibration_recording(3, per_class=2)
    path = recording.save(rec, tmp_path, "cal")
    back = recording.load(path)
    t1, z1 = ssvep.replay(rec, 2.0)
    t2, z2 = ssvep.replay(back, 2.0)
    assert np.allclose(t1, t2) and np.nanmax(np.abs(z1 - z2)) < 0.05   # samples stored as float32


# -- devices -----------------------------------------------------------------------------------------
def test_ring_keeps_the_latest_samples_and_cuts_by_time():
    r = devices.Ring(1, 100)
    for k in range(10):
        t = k * 30 + np.arange(30.0)
        r.push(t, t[:, None])
    t, x = r.view()
    assert t[-1] == 299 and np.all(np.diff(t) == 1)
    assert np.array_equal(r.view(5.0)[0], np.arange(295.0, 300.0))
    assert np.array_equal(r.view(5.0, t_end=290.0)[0], np.arange(286.0, 291.0))


def test_moving_electrodes_swaps_within_a_device_and_refuses_the_others_spot(tmp_path):
    f = devices.Feeds(("flex", "headband"), POSTERIOR, mock=True)
    f.set_position("flex", "O1", "POz")                  # T3 was at POz: they swap
    assert f.placement()["flex"] == {"O1": "POz", "O2": "PO4", "T3": "PO3", "T4": "Pz"}
    with pytest.raises(ValueError, match="Headband"):
        f.set_position("flex", "O1", "O1")
    devices.save_placement(f.placement(), tmp_path / "p.json")
    again = devices.Feeds(("flex", "headband"), POSTERIOR, mock=True,
                          placement=devices.load_placement(tmp_path / "p.json"))
    assert again.placement() == f.placement()


@pytest.fixture
def fake_sdk(monkeypatch):
    monkeypatch.syspath_prepend(FAKE)
    for mod in [m for m in sys.modules if m == "neurosdk" or m.startswith("neurosdk.")]:
        monkeypatch.delitem(sys.modules, mod)
    monkeypatch.setattr(config, "BRAINBIT_RESIST_S", 0.4)
    monkeypatch.setattr(config, "BRAINBIT_RECONNECT_S", 0.3)
    import eeg.brainbit as bb
    monkeypatch.setattr(bb, "_CLAIMS", {})
    monkeypatch.setattr(bb, "_KIND_OF", {})
    made = []

    def make(scenario, names=("flex", "headband"), devs=UNPINNED):
        monkeypatch.setenv("FAKE_BRAINBIT", scenario)
        f = devices.Feeds(names, devs)
        made.append(f)
        return f
    yield make
    for f in made:
        f.stop()


def _pump(f, seconds, until):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        f.poll()
        if until():
            return True
        time.sleep(0.01)
    return False


def test_flex_and_headband_through_the_sdk(fake_sdk):
    f = fake_sdk("dual")
    f.start()
    assert _pump(f, 15, lambda: all(d.state == "streaming" and d.samples > 300 for d in f.devices)), \
        [(d.state, d.message) for d in f.devices]
    flex, hb = f.devices
    _, x = flex.window(None, 1.0, "raw")
    amp = dict(zip(flex.positions, x.std(axis=0)))
    assert 20 < amp["PO3"] < 35                          # the fake Flex's "O1" lead (40 µV marker) sits at PO3
    _, x = hb.window(None, 1.0, "raw")
    assert 14 < dict(zip(hb.positions, x.std(axis=0)))["O2"] < 26
    assert f.measure_contact()
    assert _pump(f, 25, lambda: all(set(d.resistance) == set(d.sdk) and d.state == "streaming" for d in f.devices))


def test_a_device_dropping_out_leaves_the_other_streaming(fake_sdk):
    f = fake_sdk("dual-drop")
    f.start()
    assert _pump(f, 15, lambda: all(d.state == "streaming" for d in f.devices))
    assert _pump(f, 10, lambda: f.devices[1].state == "reconnecting")
    now = time.monotonic()
    assert not f.devices[0].silent(now)
    assert _pump(f, 15, lambda: f.devices[1].state == "streaming" and not f.devices[1].silent(time.monotonic()))


def test_missing_sdk_is_a_message_not_a_crash(monkeypatch):
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "neurosdk" else real(name, *a))
    f = devices.Feeds(("flex",), UNPINNED)
    f.start()
    assert f.state == "error" and "pyneurosdk2" in f.fatal


# -- the BrainBit worker (eeg/brainbit.py) against the fake SDK ------------------------------------------
ONE = {"bb": {"label": "BrainBit", "family": "any", "map": {"O1": "O1", "O2": "O2", "T3": "P3", "T4": "P4"}}}


def _amp(d, seconds=1.0):
    _, x = d.window(None, seconds, "raw")
    return dict(zip(d.positions, x.std(axis=0)))


@pytest.mark.parametrize("scenario,kind", [("flex4", "Flex"), ("flex1", "BrainBit Flex (v1)")])
def test_one_device_streams_microvolts_mapped_by_channel_name(fake_sdk, scenario, kind):
    f = fake_sdk(scenario, names=("bb",), devs=ONE)
    assert not f.measure_contact()                    # nothing streaming yet
    f.start()
    d = f.devices[0]
    assert _pump(f, 10, lambda: d.state == "streaming" and d.samples > 300), (d.state, d.message)
    assert kind in d.info["kind"] and d.battery == 87
    amp = _amp(d)
    # The fake Flex 4 reports channels as T3, O1, T4, O2; O1's 40 µV marker must land on "O1".
    assert max(amp, key=amp.get) == "O1" and 20 < amp["O1"] < 35     # volts were converted to µV
    assert all(2e5 < r < 4e5 for r in d.resistance.values())          # measured on connect (one device)
    assert f.measure_contact()
    assert _pump(f, 5, lambda: d.state == "resist") and _pump(f, 5, lambda: d.state == "streaming")


def test_no_device_keeps_scanning_and_stops_quickly(fake_sdk):
    f = fake_sdk("none", names=("bb",), devs=ONE)
    f.start()
    assert _pump(f, 3, lambda: f.devices[0].state == "scanning")
    t0 = time.monotonic()
    f.stop()
    assert time.monotonic() - t0 < 3.0


def test_bluetooth_drop_reconnects(fake_sdk):
    f = fake_sdk("drop", names=("bb",), devs=ONE)
    f.start()
    d = f.devices[0]
    assert _pump(f, 10, lambda: d.state == "streaming" and d.samples > 100)
    assert _pump(f, 6, lambda: d.state == "reconnecting")
    assert _pump(f, 10, lambda: d.state == "streaming")
    n = d.samples
    assert _pump(f, 5, lambda: d.samples > n + 250)                   # data flows again


def test_two_devices_start_one_at_a_time(fake_sdk):
    """The second device connects only once the first is delivering samples (a Flex v1 otherwise stalls)."""
    f = fake_sdk("dual")
    f.start()
    order = []

    def watch():
        for d in f.devices:
            if d.state in ("connecting", "streaming") and (d.name, d.state) not in order:
                order.append((d.name, d.state))
        return all(d.state == "streaming" for d in f.devices)
    assert _pump(f, 15, watch)
    first = order[0][0]
    other = next(n for n, _ in order if n != first)
    assert order.index((first, "streaming")) < order.index((other, "streaming"))
    assert not any(d.state == "resist" for d in f.devices)


@pytest.mark.parametrize("pinned", [True, False])
def test_flex_v1_and_headband_look_alike_but_land_in_the_right_slots(fake_sdk, pinned):
    """Both advertise as LEBrainBit "BrainBit"; only the firmware tells a Flex v1 from the headband.
    Unpinned, the Flex slot meets the (stronger) headband first, hands it over, and finds the Flex."""
    devs = {n: dict(d) for n, d in POSTERIOR.items()} if pinned else \
        {n: {k: v for k, v in d.items() if k != "serial"} for n, d in POSTERIOR.items()}
    f = fake_sdk("dual-v1", devs=devs)
    f.start()
    assert _pump(f, 20, lambda: all(d.state == "streaming" and d.samples > 300 for d in f.devices)), \
        [(d.state, d.message) for d in f.devices]
    assert {d.name: d.info.get("kind") for d in f.devices} == {"flex": "BrainBit Flex (v1)",
                                                                "headband": "BrainBit headband"}
    assert 20 < _amp(f.devices[0])["PO3"] < 35 and 14 < _amp(f.devices[1])["O2"] < 26


# -- look-away calibration and the stop gate -----------------------------------------------------------
def test_look_away_protocol_runs_rover_then_cross_then_box_and_the_scope_keeps_its_pauses():
    from eeg import cues

    p = cues.Protocol(classes=("up", "rest"), per_class=2, practice=0, away_s=4.0, baseline_s=3.0, cue_s=5.0,
                      pause_s=(0.0, 0.0), break_every=0, lead_in_s=1.0, tail_s=1.0)
    kinds = [s.kind for s in cues.plan(p, np.random.default_rng(0))]
    assert kinds == ["ready"] + ["away", "fixation", "cue"] * 4 + ["done"]
    run = cues.Run(p, 0.0, seed=0)
    for k in range(600):
        run.update(k * 0.1)
    tr = run.trials
    assert len(tr) == 4 and all(t["away"] < t["baseline"] < t["cue"] < t["end"] for t in tr)
    assert all(abs(t["baseline"] - t["away"] - 4.0) < 0.11 for t in tr)
    scope = cues.plan(cues.Protocol(), np.random.default_rng(0))        # the scope's motor sessions: as before
    assert "away" not in {s.kind for s in scope} and "pause" in {s.kind for s in scope}


def test_calibration_trials_count_the_look_away_as_rest():
    meta = {"trials": [{"label": "up", "away": 0.0, "baseline": 4.0, "cue": 7.0, "end": 12.0},
                       {"label": "left", "away": 12.0, "cue": 16.0, "end": 21.0}],     # (no 3-2-1 hand-over)
            "protocol": {"away_s": 4.0}}
    tr = ssvep.calibration_trials(meta)
    assert [(t.label, t.onset, t.end, t.prev) for t in tr] == [       # the 3-2-1 (4-7 s) is left out
        ("rest", 0.0, 4.0, "rest"), ("up", 7.0, 12.0, "rest"), ("rest", 12.0, 16.0, "up"), ("left", 16.0, 21.0, "rest")]
    p = ssvep.calibration_protocol()
    assert p.away_s == config.SSVEP_CAL_AWAY_S > 0 and "rest" not in p.classes
    kinds = [s.kind for s in __import__("eeg.cues", fromlist=["plan"]).plan(p, np.random.default_rng(0))]
    assert kinds[kinds.index("away"):kinds.index("away") + 4] == ["away", "fixation", "cue", "away"]
    assert "pause" not in kinds
    # rested at the rover vs at the centre: not combined
    assert ssvep.rest_style(meta) == "rover" and ssvep.rest_style({"protocol": {"away_s": 0.0}}) == "centre"
    assert ssvep.rest_style({"segments": []}) == "centre"


def _motion_feed(mc, rng, seconds, t0, x_uv=5.0, m_uv=3.0, spikes=(), muscle=1.0):
    """Feed MotionCheck 0.5 s windows every 0.1 s; ``spikes``: channels with big samples, ``muscle``: muscle x."""
    states = []
    for k in range(int(seconds * 10)):
        x = rng.normal(0, x_uv, (125, 4))
        m = rng.normal(0, m_uv, (125, 4))
        for c in spikes:
            x[::3, c] += 60.0                       # a third of the samples far out: an electrode shifting
        m *= muscle
        states.append(mc.update(t0 + k * 0.1, x, m))
    return states


def test_motion_check_stops_for_movement_and_muscle_but_not_one_loose_electrode():
    from eeg.clean import MOVING, MUSCLE, MotionCheck

    rng = np.random.default_rng(0)
    mc = MotionCheck(4)
    assert set(_motion_feed(mc, rng, 10, 0.0)) == {None}                 # sitting still
    assert set(_motion_feed(mc, rng, 1, 10.0, spikes=(3,))[2:]) == {None}   # one electrode popping: down-weighted
    assert set(_motion_feed(mc, rng, 1, 11.0, spikes=(0, 1, 2))) == {MOVING}  # the head moving: most of them
    assert set(_motion_feed(mc, rng, 1, 12.0, muscle=6.0)) == {MUSCLE}        # a clenched jaw
    assert set(_motion_feed(mc, rng, 1, 13.0)) == {None}                      # calm again at once (gate holds)
    # a lasting change (an electrode reseated noisier) stops it only until the calm levels are relearned
    states = _motion_feed(mc, rng, 15, 14.0, muscle=6.0)
    assert states[0] == MUSCLE and set(states[-30:]) == {None}


def test_stop_gate_holds_after_the_last_disturbance():
    g = ssvep.StopGate(hold_s=1.0)
    assert g.update(0.0, [None, None]) is None
    assert g.update(0.1, [None, "muscle"]) == "muscle"
    assert g.update(1.0, [None]) == "muscle"
    assert g.update(1.2, []) is None


def test_simulated_smoother_stops_at_once_on_a_forced_window():
    t = np.arange(40) * 0.1
    lab = np.zeros(40, dtype=int)                    # box 0 throughout...
    forced = np.zeros(40, dtype=bool)
    forced[20] = True
    lab[20] = 3                                      # ...but the stop gate at window 20
    cmd = ssvep.simulate(t, lab, 5, rest=3, forced=forced)
    assert cmd[19] == 0 and cmd[20] == 3             # stopped at once, no debounce
    assert list(cmd[21:25]) == [3] * 4 and cmd[25] == 0   # then a fresh full streak to drive again


def test_replay_marks_a_jaw_clench_for_the_stop_gate():
    clk = Clock()
    f = devices.Feeds(("flex",), POSTERIOR, mock=True, mock_preset="clean", clock=clk, seed=3)
    f.start()
    f.start_recording()
    for k in range(240):                             # 12 s, the jaw clenched from 8 to 10 s
        clk.t += 0.05
        f.mock_input(gaze="up", jaw=8.0 <= k * 0.05 < 10.0)
        f.poll()
    rec = f.stop_recording()
    f.stop()
    t, z, stop = ssvep.replay(rec, 2.0, with_stop=True)
    rel = t - t[0] + 2.0
    assert not stop[rel < 7.9].any()
    assert stop[(rel > 8.3) & (rel < 10.0)].all()
    assert not stop[rel > 11.5].any()                # released (1 s hold)


def test_a_box_whose_frequency_changed_starts_from_zero(tmp_path):
    rec = _calibration_recording(1)
    rec["meta"]["frames"] = dict(config.SSVEP_FRAMES)
    same = ssvep.fit(rec, windows=(2.0,))
    old = {**config.SSVEP_FRAMES, "left": config.SSVEP_FRAMES["left"] + 1}    # ← flickered at another rate then
    rec["meta"]["frames"] = old
    other = ssvep.fit(rec, windows=(2.0,))
    n_left = sum(t.label == "left" for t in ssvep.calibration_trials(rec["meta"]))
    assert other.report["trials"] == same.report["trials"] - n_left > 0          # its trials are left out
    # settings saved for other frequencies aren't loaded
    same.setup = "flex:PO3"
    ssvep.save_settings(same, tmp_path)
    assert ssvep.load_settings("default", "flex:PO3", tmp_path)[0] is not None
    same.frames = old
    ssvep.save_settings(same, tmp_path)
    s, msg = ssvep.load_settings("default", "flex:PO3", tmp_path)
    assert s is None and "other flicker frequencies" in msg
