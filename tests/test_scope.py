"""neurover scope (python scope.py): per-electrode metrics, recordings, live feeds (mock and the fake
BrainBit SDK), and a headless run of the window."""
import math
import sys
import time
from pathlib import Path

import numpy as np
import pygame
import pytest

import config
from eeg import recording
from eeg.devices import Feeds, Ring
from rawscope import analysis

FS = 250.0
FAKE = str(Path(__file__).parent / "fake_neurosdk")
UNPINNED = {n: {k: v for k, v in d.items() if k != "serial"} for n, d in config.EEG_DEVICES.items()}


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


# -- analysis -------------------------------------------------------------------------------
def _signal(seconds=60.0, seed=0):
    t = np.arange(int(seconds * FS)) / FS
    rng = np.random.default_rng(seed)
    f = analysis.TARGETS["left"]
    x = 50_000 + 10 * np.sin(2 * np.pi * f * t) + 5 * np.sin(2 * np.pi * 60 * t) + rng.normal(0, 2, len(t))
    return t, x


def test_metrics_of_a_known_signal():
    t, x = _signal()
    s = analysis.analyse(t, x, FS, resistance=4e5, duration_s=60.0)
    assert 6.8 < s["rms"] < 7.6                      # 10 µV sine = 7.07 µV RMS, plus a little noise
    assert s["line"] == pytest.approx(5 / math.sqrt(2), rel=0.05)
    assert s["snr_left"] > 100 and s["snr_left"] > 50 * s["snr_right"]   # the ← box's frequency
    assert s["dc"] == pytest.approx(50.0, abs=0.1)   # mV
    assert s["resist"] == 400 and s["dropout"] == pytest.approx(0, abs=0.1) and s["artifact"] == 0
    assert analysis.status(s) == ("OK", "good")


def test_gaps_split_the_stream_and_count_as_missing():
    t, x = _signal()
    keep = (t < 20) | (t >= 26)                      # a 6 s Bluetooth gap
    s = analysis.analyse(t[keep], x[keep], FS, duration_s=60.0)
    assert s["dropout"] == pytest.approx(10.0, abs=0.1)
    assert s["artifact"] == 0                        # the jump across the gap is not filtered as a step
    assert s["line"] == pytest.approx(5 / math.sqrt(2), rel=0.05)


def test_artifacts_are_counted_but_kept_out_of_the_spectrum():
    t, x = _signal()
    x = x.copy()
    x[3000:3100] += 400                              # an electrode pop
    s = analysis.analyse(t, x, FS)
    used, total = s["segments"]
    assert used < total and s["artifact"] > 0
    assert s["line"] == pytest.approx(5 / math.sqrt(2), rel=0.05)


def test_muscle_noise_is_not_mains():
    t = np.arange(int(30 * FS)) / FS
    rng = np.random.default_rng(1)
    from scipy import signal
    emg = signal.sosfilt(signal.butter(4, [20, 100], "bandpass", fs=FS, output="sos"), rng.normal(0, 30, len(t)))
    assert analysis.analyse(t, emg, FS)["line"] < 1.0


@pytest.mark.parametrize("x, expect", [(np.full(2500, 12.0), "FLAT"), (np.full(2500, 400_000.0), "SATURATED")])
def test_status_flags_dead_electrodes(x, expect):
    t = np.arange(len(x)) / FS
    assert analysis.status(analysis.analyse(t, x, FS))[0] == expect


def test_ring_keeps_the_latest_samples():
    r = Ring(2, 100)
    for k in range(10):
        t = k * 30 + np.arange(30.0)
        r.push(t, np.column_stack([t, -t]))
    t, x = r.view()
    assert len(t) >= 100 and t[-1] == 299 and np.all(np.diff(t) == 1) and np.all(x[:, 1] == -t)
    assert np.array_equal(r.view(9.5)[0], np.arange(290.0, 300.0))


# -- recordings -----------------------------------------------------------------------------
def _recording(tmp_path, seconds=6.0, mock=True, seed=0):
    clock = Clock()
    feeds = Feeds(("flex", "headband"), config.EEG_DEVICES, mock=mock, clock=clock, seed=seed)
    feeds.start()
    feeds.poll()
    feeds.start_recording()
    for _ in range(int(seconds / 0.04)):
        clock.t += 0.04
        feeds.poll()
    return feeds, feeds.stop_recording()


def test_mock_feeds_record_every_electrode_of_both_devices(tmp_path):
    feeds, rec = _recording(tmp_path)
    assert [d.state for d in feeds.devices] == ["streaming", "streaming"]
    assert [n.position for n in feeds.nodes] == ["PO3", "POz", "PO4", "Pz", "O1", "O2", "T3", "T4"]
    assert rec["meta"]["duration_s"] == pytest.approx(6.0)
    for name in ("flex", "headband"):
        d = rec["devices"][name]
        assert d["x"].shape == (1500, 4) and np.all(np.diff(d["t"]) > 0)
        assert d["positions"] == list(config.EEG_DEVICES[name]["map"].values())


def test_recording_round_trip_rename_and_listing(tmp_path):
    feeds, rec = _recording(tmp_path)
    feeds.devices[0].resistance = {"O1": 3e5}
    path = recording.save(rec, tmp_path)
    assert path.name.endswith(".npz") and "_" in path.stem
    back = recording.load(path)
    for name, d in rec["devices"].items():
        assert np.array_equal(back["devices"][name]["t"], d["t"])
        assert np.allclose(back["devices"][name]["x"], d["x"], atol=0.05)   # stored as float32
    renamed = recording.rename(path, "eyes closed / flex@Oz")
    assert not path.exists() and renamed.stem.endswith("_eyes-closed-flex-Oz")
    assert recording.info(renamed)["name"] == "eyes closed / flex@Oz"
    second = recording.save(_recording(tmp_path, seconds=2.0)[1], tmp_path, "later")
    listed = recording.list_recordings(tmp_path)
    assert {m["path"] for m in listed} == {renamed, second}
    stats = analysis.analyse_recording(recording.load(renamed))
    assert set(stats) == {"PO3", "POz", "PO4", "Pz", "O1", "O2", "T3", "T4"}
    assert stats["PO3"]["label"] == "Flex" and stats["PO3"]["sdk"] == "O1"


def test_mock_contact_measurement_pauses_then_reports_every_electrode():
    clock = Clock()
    feeds = Feeds(("flex", "headband"), config.EEG_DEVICES, mock=True, clock=clock, seed=0)
    feeds.start()
    feeds.poll()
    assert feeds.measure_contact()
    feeds.poll()
    assert all(d.state == "resist" for d in feeds.devices)
    clock.t += 1.5
    feeds.poll()
    assert all(d.state == "streaming" and set(d.resistance) == set(d.sdk) for d in feeds.devices)


def test_report_prints_every_electrode_and_metric(tmp_path):
    import scope

    a = recording.save(_recording(tmp_path, seed=1)[1], tmp_path, "a")
    b = recording.save(_recording(tmp_path, seed=2)[1], tmp_path, "b")
    text = scope.report([a, b])
    for m in analysis.METRICS:
        assert m.label in text
    assert text.count("  PO3 ") == len(analysis.METRICS)


# -- the real worker, against the fake SDK ---------------------------------------------------
@pytest.fixture
def fake_sdk(monkeypatch):
    monkeypatch.syspath_prepend(FAKE)
    for mod in [m for m in sys.modules if m == "neurosdk" or m.startswith("neurosdk.")]:
        monkeypatch.delitem(sys.modules, mod)
    monkeypatch.setattr(config, "BRAINBIT_RESIST_S", 0.4)
    monkeypatch.setattr(config, "BRAINBIT_RECONNECT_S", 0.3)
    import eeg.brainbit as bw
    monkeypatch.setattr(bw, "_CLAIMS", {})
    monkeypatch.setattr(bw, "_KIND_OF", {})
    monkeypatch.setenv("FAKE_BRAINBIT", "dual")
    feeds = Feeds(("flex", "headband"), UNPINNED)
    yield feeds
    feeds.stop()


def _pump(feeds, seconds, until):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        feeds.poll()
        if until():
            return True
        time.sleep(0.01)
    return False


def test_flex_and_headband_raw_feeds_and_recording_via_the_sdk(fake_sdk):
    feeds = fake_sdk
    feeds.start()
    assert _pump(feeds, 15, lambda: all(d.state == "streaming" and d.samples > 250 for d in feeds.devices)), \
        [(d.state, d.message) for d in feeds.devices]
    feeds.start_recording()
    assert _pump(feeds, 5, lambda: all(n > 250 for n in feeds.recorder.samples.values())
                 and len(feeds.recorder.samples) == 2)
    rec = feeds.stop_recording()
    flex, hb = rec["devices"]["flex"], rec["devices"]["headband"]
    # Flex lead "O1" (40 µV 5 Hz marker) sits at PO3; the headband's O2 has a 30 µV marker.
    amp_flex = dict(zip(flex["positions"], flex["x"].std(axis=0)))
    amp_hb = dict(zip(hb["positions"], hb["x"].std(axis=0)))
    assert 20 < amp_flex["PO3"] < 35 and 14 < amp_hb["O2"] < 26
    assert flex["kind"] and hb["kind"] and flex["battery"] is not None
    # R: both pause, are measured one at a time, then stream again.
    assert feeds.measure_contact()
    assert _pump(feeds, 25, lambda: all(set(d.resistance) == set(d.sdk) and d.state == "streaming"
                                        for d in feeds.devices))


# -- the window, headless ---------------------------------------------------------------------
def test_window_records_names_and_compares(tmp_path):
    from rawscope.app import ScopeApp

    feeds = Feeds(("flex", "headband"), config.EEG_DEVICES, mock=True, seed=0)
    app = ScopeApp(feeds, tmp_path)
    steps = {"phase": 0}

    def script(a, t):
        if steps["phase"] == 0 and t > 0.3:
            a.toggle_recording()
            steps["phase"] = 1
        elif steps["phase"] == 1 and t > 2.0:
            a.toggle_recording()                     # stop: asks for a name
            for ch in "eyes closed":
                pygame.event.post(pygame.event.Event(pygame.TEXTINPUT, text=ch))
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, mod=0, unicode="\r",
                                                 scancode=0))
            steps["phase"] = 2
        elif steps["phase"] == 2 and t > 2.3:
            a.toggle_recording()
            steps["phase"] = 3
        elif steps["phase"] == 3 and t > 3.5:       # the second recording is still running
            a.set_view("compare")
            assert [m["name"] for m in a.library] == ["eyes closed"]
            a.toggle_selected(a.library[0]["path"])
            assert a.stats_for(a.library[0])
            for i in range(len(analysis.METRICS)):   # every metric draws
                a.metric_i = i
                a.draw()
            steps["phase"] = 4

    shot = tmp_path / "shot.png"
    app.run(duration=4.0, screenshot=str(shot), script=script)
    assert shot.exists() and steps["phase"] == 4
    # The second recording was still running at exit: it was saved, not lost; the selection survived.
    names = sorted(m.get("name", "") for m in recording.list_recordings(tmp_path))
    assert names == ["", "eyes closed"]
    assert app.view == "compare" and [p.stem.endswith("eyes-closed") for p in app.selected] == [True]


# -- moving electrodes ------------------------------------------------------------------------
from rawscope import placement  # noqa: E402
from rawscope.playback import Playback  # noqa: E402


def test_reassign_swaps_within_a_device_and_refuses_another_devices_spot():
    pos = {"flex": ["PO3", "PO4", "POz", "Pz"], "headband": ["O1", "O2", "T3", "T4"]}
    assert placement.reassign(pos, "flex", 0, "Oz")["flex"] == ["Oz", "PO4", "POz", "Pz"]
    assert placement.reassign(pos, "flex", 0, "PO4")["flex"] == ["PO4", "PO3", "POz", "Pz"]   # swap
    with pytest.raises(ValueError, match="Headband"):
        placement.reassign(pos, "flex", 0, "O1", {"headband": "Headband"})
    with pytest.raises(ValueError):
        placement.reassign(pos, "flex", 0, "  ")
    assert pos["flex"] == ["PO3", "PO4", "POz", "Pz"]             # input untouched


def test_moving_an_electrode_live_changes_names_not_rows_and_is_recorded(tmp_path):
    clock = Clock()
    feeds = Feeds(("flex", "headband"), config.EEG_DEVICES, mock=True, clock=clock, seed=0)
    feeds.start()
    feeds.poll()
    rows = [(n.device, n.sdk) for n in feeds.nodes]
    feeds.set_position("flex", "O1", "Oz")
    feeds.set_position("flex", "T3", "Pz")                          # takes T4's spot: they swap
    assert feeds.placement()["flex"] == {"O1": "Oz", "O2": "PO4", "T3": "Pz", "T4": "POz"}
    assert [(n.device, n.sdk) for n in feeds.nodes] == rows          # rows stay put
    assert [n.position for n in feeds.nodes][:4] == ["Oz", "Pz", "PO4", "POz"]
    with pytest.raises(ValueError, match="Headband"):
        feeds.set_position("flex", "O1", "O2")
    feeds.start_recording()
    with pytest.raises(ValueError, match="recording"):
        feeds.set_position("flex", "O1", "Iz")
    clock.t += 2.0
    feeds.poll()
    rec = feeds.stop_recording()
    assert rec["devices"]["flex"]["positions"] == ["Oz", "PO4", "Pz", "POz"]
    # Remembered: a new Feeds starts where we left off; one that would double up a spot is ignored.
    placement.save(feeds.placement(), tmp_path / "p.json")
    again = Feeds(("flex", "headband"), config.EEG_DEVICES, mock=True, placement=placement.load(tmp_path / "p.json"))
    assert again.placement() == feeds.placement() and not again.placement_note
    bad = Feeds(("flex", "headband"), config.EEG_DEVICES, mock=True, placement={"flex": {"O1": "O1", "O2": "PO4",
                                                                                          "T3": "POz", "T4": "Pz"}})
    assert bad.devices[0].positions == bad.devices[0].config_positions and bad.placement_note
    assert placement.load(tmp_path / "missing.json") == {}


def test_relabelling_a_saved_recording_moves_names_and_resistance(tmp_path):
    feeds, rec = _recording(tmp_path)
    rec["devices"]["flex"]["resistance"] = {"PO3": 3e5, "PO4": 9e5}
    path = recording.save(rec, tmp_path, "oops")
    new = recording.relabel(path, "flex", "O1", "PO4")               # O1 and O2 were plugged in swapped
    assert new == path
    d = recording.load(path)["devices"]["flex"]
    assert d["positions"] == ["PO4", "PO3", "POz", "Pz"]
    assert d["resistance"] == {"PO4": 3e5, "PO3": 9e5}
    assert np.allclose(d["x"], rec["devices"]["flex"]["x"], atol=0.05)   # samples untouched
    with pytest.raises(ValueError):
        recording.relabel(path, "flex", "O1", "T3")                  # the headband's


# -- playback ---------------------------------------------------------------------------------
def test_playback_cuts_at_the_playhead_and_runs_at_its_speed(tmp_path):
    feeds, rec = _recording(tmp_path, seconds=6.0)
    hb = rec["devices"]["headband"]
    keep = (hb["t"] - rec["meta"]["start"] < 2.0) | (hb["t"] - rec["meta"]["start"] > 3.5)  # a Bluetooth drop
    hb["t"], hb["x"] = hb["t"][keep], hb["x"][keep]
    pb = Playback(recording.load(recording.save(rec, tmp_path)))
    assert pb.duration == pytest.approx(6.0, abs=0.01)
    assert [n.position for n in pb.nodes] == ["PO3", "POz", "PO4", "Pz", "O1", "O2", "T3", "T4"]
    flex, head = pb.devices
    pb.seek(4.0)
    t, x = flex.window(pb.head, 2.0, "raw")
    assert t[-1] <= pb.head and t[0] > pb.head - 2.0 and len(t) == pytest.approx(500, abs=2)
    assert flex.window(pb.head, 2.0, "display")[1].shape == x.shape
    pb.seek(3.0)
    assert head.silent(pb.head) and not flex.silent(pb.head)          # inside the headband's gap
    assert len(head.coverage) == 2
    pb.seek(0.0)
    pb.speed_i = pb.speed_i + 1                                       # 2x
    pb.toggle(100.0)
    pb.update(101.0)
    assert pb.pos == pytest.approx(2.0)
    pb.update(110.0)
    assert pb.pos == pb.duration and not pb.playing                   # stops at the end...
    pb.toggle(111.0)
    assert pb.pos == 0.0 and pb.playing                               # ...and plays again from the start


def test_window_moves_electrodes_and_plays_back(tmp_path):
    from rawscope.app import ScopeApp

    feeds = Feeds(("flex", "headband"), config.EEG_DEVICES, mock=True, seed=0)
    app = ScopeApp(feeds, tmp_path, placement_path=tmp_path / "placement.json")
    recording.save(_recording(tmp_path, seconds=4.0)[1], tmp_path, "first")
    steps = {"phase": 0}

    def key(k):
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=k, mod=0, unicode="", scancode=0))

    def script(a, t):
        p = steps["phase"]
        if p == 0 and t > 0.3:
            a.open_picker("live", "flex", "O1")
            a.draw()
            assert a.apply_picker("Oz")
            assert placement.load(tmp_path / "placement.json")["flex"]["O1"] == "Oz"
            a.open_picker("live", "flex", "O1")
            assert not a.apply_picker("O2") and a.picker is not None     # the headband's: refused, stays open
            a.picker = None
            a.set_view("compare")
            steps["phase"] = 1
        elif p == 1 and t > 0.5:
            key(pygame.K_p)                                               # play the focused recording
            steps["phase"] = 2
        elif p == 2 and t > 1.2:
            assert a.view == "playback" and a.playback.playing and a.playback.pos > 0.3
            key(pygame.K_SPACE)                                           # pause
            steps["phase"] = 3
        elif p == 3 and t > 1.4:
            assert not a.playback.playing
            a.open_picker("playback", "headband", "T3")
            a.draw()
            assert a.apply_picker("Cz")                                   # fix the recording
            assert a.playback.devices[1].positions == ["O1", "O2", "Cz", "T4"]
            key(pygame.K_TAB)
            steps["phase"] = 4
        elif p == 4 and t > 1.6:
            assert a.view == "live"
            steps["phase"] = 5

    app.run(duration=1.8, script=script)
    assert steps["phase"] == 5
    saved = recording.load(recording.list_recordings(tmp_path)[0]["path"])
    assert saved["devices"]["headband"]["positions"] == ["O1", "O2", "Cz", "T4"]


# -- cued trials: protocol ----------------------------------------------------------------------
from eeg import cues  # noqa: E402
from rawscope import trials  # noqa: E402


@pytest.mark.parametrize("seed", range(5))
def test_trial_order_is_balanced_and_never_three_in_a_row(seed):
    order = cues.sequence(cues.CLASSES, 15, 2, np.random.default_rng(seed))
    assert sorted(order) == sorted(list(cues.CLASSES) * 15)
    assert all(not (order[i] == order[i + 1] == order[i + 2]) for i in range(len(order) - 2))
    assert order != cues.sequence(cues.CLASSES, 15, 2, np.random.default_rng(seed + 100))   # actually random


def test_session_plan_has_practice_breaks_and_jittered_pauses():
    p = cues.Protocol(per_class=15)
    steps = cues.plan(p, np.random.default_rng(0))
    kinds = [s.kind for s in steps]
    assert kinds[0] == "ready" and kinds[-1] == "done" and kinds.count("go") == 1
    cue = [s for s in steps if s.kind == "cue"]
    assert len(cue) == p.practice + 45 and sum(s.practice for s in cue) == p.practice
    assert [s.trial for s in cue if not s.practice] == list(range(45))
    assert kinds.count("break") == 2                                    # after trials 15 and 30, not after 45
    after = [steps[i - 1].trial for i, k in enumerate(kinds) if k == "break"]
    assert after == [14, 29]
    pauses = [s.seconds for s in steps if s.kind == "pause"]
    assert all(1.5 <= x <= 3.0 for x in pauses) and np.std(pauses) > 0.2
    assert all(s.seconds == 2.0 for s in steps if s.kind == "fixation") and all(s.seconds == 4.0 for s in cue)


def test_run_marks_each_trial_and_ending_early_keeps_the_finished_ones():
    p = cues.Protocol(per_class=4, practice=2, break_every=6)
    run = cues.Run(p, start=100.0, name="x", seed=3)
    beeps, t = [], 100.0
    while not run.finished and t < 100.0 + 15.0 + 8 * 4.0:              # partway: lead-in, practice, a few trials
        t += 1 / 60
        beeps += [s for s in run.update(t) if s.kind == "cue"]
    assert run.trials and run.trials[0].get("practice") and len(beeps) >= len(run.trials)
    for tr in run.trials:
        assert tr["cue"] - tr["baseline"] == pytest.approx(p.baseline_s, abs=0.02)
        assert tr["end"] - tr["cue"] == pytest.approx(p.cue_s, abs=0.02)
    labels = [s.label for s in run.steps if s.kind == "cue"][:len(run.trials)]
    assert [tr["label"] for tr in run.trials] == labels
    done = len(run.trials)
    while run.step.kind != "cue":                                        # stop in the middle of a cue
        t += 1 / 60
        run.update(t)
    run.stop()
    assert run.finished and run.ended_early and len(run.trials) == done
    meta = run.meta()
    assert meta["protocol"]["task"] == "imagine" and meta["protocol"]["ended_early"]
    assert meta["protocol"]["completed"] == sum(1 for x in meta["trials"] if not x.get("practice"))
    # breaks (and the lead-in) can be cut short; nothing else can
    run2 = cues.Run(p, start=0.0, seed=1)
    assert run2.step.kind == "ready" and run2.skip(0.5)
    assert [s.kind for s in run2.update(0.5)][:1] == ["fixation"] and not run2.skip(0.6)


# -- cued trials: recordings ----------------------------------------------------------------------
def _cued_mock(per_class=15, seed=0, hand=True, placement=None, step=0.05, protocol=None):
    """A cued session recorded from the mock (its virtual motor cortex desynchronises with the cue)."""
    clock = Clock()
    feeds = Feeds(("flex", "headband"), config.EEG_DEVICES, mock=True, mock_motor=True, clock=clock, seed=seed,
                  placement=placement or {"flex": {"O1": "C3", "O2": "C4", "T3": "Cz", "T4": "Fz"}})
    feeds.start()
    feeds.poll()
    feeds.start_recording()
    run = cues.Run(protocol or cues.Protocol(per_class=per_class), clock(), seed=seed)
    while not run.finished:
        clock.t += step
        run.update(clock())
        feeds.mock_cue(run.step.label if hand and run.step.kind == "cue" and not run.finished else None)
        feeds.poll()
    rec = feeds.stop_recording()
    rec["meta"].update(run.meta())
    return rec


def test_trial_markers_survive_save_rename_relabel_and_playback(tmp_path):
    rec = _cued_mock(protocol=cues.Protocol(per_class=1, practice=1, lead_in_s=1.0))
    path = recording.save(rec, tmp_path, "cued")
    back = recording.load(path)["meta"]
    assert back["trials"] == rec["meta"]["trials"] and back["protocol"]["task"] == "imagine"
    t = rec["devices"]["flex"]["t"]
    assert all(t[0] < tr["baseline"] < tr["cue"] < tr["end"] < t[-1] for tr in back["trials"])   # the samples' clock
    assert trials.has_trials(back) and len(back["trials"]) == 4                                     # 1 practice + 3
    path = recording.rename(path, "renamed")
    path = recording.relabel(path, "flex", "T4", "FC3")
    assert recording.load(path)["meta"]["trials"] == rec["meta"]["trials"]
    pb = Playback(recording.load(path))
    first = pb.trials[1]                                                 # the first real trial
    assert pb.cue_at(first["cue"] + 0.1)[0] == 1 and pb.cue_at(first["baseline"]) is None
    # A recording from before cued trials: nothing changes.
    old = _recording(tmp_path)[1]
    assert not trials.has_trials(old["meta"]) and Playback(recording.load(recording.save(old, tmp_path))).trials == []


def _synthetic_trials(planted=True, per_class=20, seed=0, fs=FS):
    """A Flex at C3 C4 Cz Fz: white noise + a 10.5 Hz mu rhythm per hemisphere, one trial every 8 s.
    ``planted``: the rhythm halves over the opposite hemisphere while a hand is cued. Trials 3, 8, 13...
    have a big movement artifact common to every lead, trial 5 a pop on C3 alone, trial 9 a gap."""
    rng = np.random.default_rng(seed)
    labels = cues.sequence(cues.CLASSES, per_class, 2, rng)
    start, lead, every = 500.0, 4.0, 8.0
    t = start + np.arange(int((lead + every * len(labels) + 4.0) * fs)) / fs
    gain = np.ones((len(t), 2))
    marks = []
    for i, lab in enumerate(labels):
        cue = start + lead + i * every + 2.0
        marks.append({"label": lab, "baseline": cue - 2.0, "cue": cue, "end": cue + 4.0})
        if planted and lab in ("left", "right"):
            gain[(t >= cue + 0.3) & (t < cue + 4.0), 1 if lab == "left" else 0] = 0.5
    phase = 2 * np.pi * np.cumsum(10.5 + 0.3 * rng.standard_normal((len(t), 2)), axis=0) / fs
    x = rng.normal(0, 10.0, (len(t), 4)) + 1000.0 * np.arange(1, 5)        # noise + each lead's DC offset
    x[:, :2] += 3.0 * np.sqrt(2) * np.sin(phase) * gain
    for i in range(3, len(labels), 5):
        sel = (t > marks[i]["cue"] + 1.0) & (t < marks[i]["cue"] + 1.6)
        x[sel] += 400.0 * np.sin(np.pi * (t[sel] - t[sel][0]) / 0.6)[:, None]
    x[np.argmin(np.abs(t - marks[5]["cue"] - 2.0)), 0] += 500.0
    keep = ~((t > marks[9]["cue"] + 1.0) & (t < marks[9]["cue"] + 1.5))
    dev = {"name": "flex", "label": "Flex", "sdk": ["O1", "O2", "T3", "T4"], "positions": ["C3", "C4", "Cz", "Fz"],
           "t": t[keep], "x": x[keep]}
    meta = {"format": recording.FORMAT, "name": "synthetic", "created": 0.0, "start": start, "end": float(t[-1]),
            "duration_s": float(t[-1] - start), "fs": fs, "mock": True, "events": [], "trials": marks,
            "protocol": {**cues.Protocol(per_class=per_class).to_meta(), "task": "imagine"}}
    return {"meta": meta, "devices": {"flex": dev}}, labels


def test_trial_analysis_finds_a_planted_erd():
    rec, labels = _synthetic_trials(planted=True)
    res = trials.analyse(rec, "device")
    e = {x["position"]: x for x in res["electrodes"]}
    assert [x["position"] for x in res["electrodes"]][:3] == ["C3", "Cz", "C4"]
    mu = lambda pos, c: e[pos]["erd"]["mu"][c]                               # noqa: E731
    assert mu("C3", "right")["mean"] < -2.0 and mu("C3", "right")["p"] < 1e-3      # right hand: left hemisphere
    assert mu("C4", "left")["mean"] < -2.0 and mu("C4", "left")["p"] < 1e-3        # left hand: right hemisphere
    assert abs(mu("C3", "rest")["mean"]) < 1.0 and abs(mu("C4", "rest")["mean"]) < 1.0
    assert e["C3"]["lr"]["mu"]["p"] < 1e-3
    lat = res["lateralisation"]["mu"]
    assert lat["by_class"]["left"]["mean"] < -1.5 < 1.5 < lat["by_class"]["right"]["mean"] and lat["lr"]["p"] < 1e-4
    # the time course: flat before the cue, down in the task window
    mean, _ = e["C3"]["course"]["mu"]["right"]
    times = res["times"]
    assert np.nanmean(mean[(times > 1.0) & (times < 3.5)]) < -2.0
    assert abs(np.nanmean(mean[(times > -1.5) & (times < -0.5)])) < 0.7
    c = res["classifier"]["flex"]
    assert c["classes"] == ["left", "right"] and c["acc"] >= 0.85 and c["p"] < 0.02
    assert 0.4 < c["chance"] < 0.6 and len(c["null"]) == trials.N_PERM


def test_trial_analysis_finds_nothing_when_nothing_is_planted():
    rec, _ = _synthetic_trials(planted=False, seed=1)
    res = trials.analyse(rec, "device")
    for e in res["electrodes"]:
        for band in e["erd"].values():
            assert all(abs(s["mean"]) < 1.0 for s in band.values())
    assert res["lateralisation"]["mu"]["lr"]["p"] > 0.05
    c = res["classifier"]["flex"]
    assert c["p"] > 0.05 and c["acc"] < 0.7 and 0.4 < c["chance"] < 0.6


def test_artifacts_are_dropped_on_the_chosen_reference_and_gaps_count_as_missing():
    rec, labels = _synthetic_trials()
    n_common = len(range(3, len(labels), 5))
    dev = trials.analyse(rec, "device", n_perm=10)["devices"]["flex"]
    raw = trials.analyse(rec, "raw", n_perm=10)["devices"]["flex"]
    # The movement artifact is the same on every lead: the device average removes it, as recorded it's dropped.
    assert (dev["artifact"], dev["missing"], dev["kept"]) == (1, 1, len(labels) - 2)
    assert (raw["artifact"], raw["missing"]) == (1 + n_common, 1)
    assert "dropped" in trials.kept_text(raw)
    x = rec["devices"]["flex"]["x"]
    assert np.allclose(trials.reference(x, "device").mean(axis=1), 0) and trials.reference(x, "raw") is x


def test_mock_cued_session_shows_its_planted_erd():
    rec = _cued_mock(per_class=15, seed=0)
    res = trials.analyse(rec, "device", n_perm=100)
    e = {x["position"]: x for x in res["electrodes"]}
    assert e["C3"]["erd"]["mu"]["right"]["mean"] < -1.0 and e["C3"]["erd"]["mu"]["right"]["p"] < 0.01
    assert res["lateralisation"]["mu"]["lr"]["p"] < 0.01
    flex, head = res["classifier"]["flex"], res["classifier"]["headband"]
    assert flex["acc"] > flex["chance"] + 0.15 and head["p"] > 0.05      # the headband (O1/O2/T3/T4) sees nothing
    # no cue given to the virtual subject: no effect
    calm = trials.analyse(_cued_mock(per_class=15, seed=0, hand=False), "device", n_perm=100)
    assert calm["lateralisation"]["mu"]["lr"]["p"] > 0.05


def test_mock_motor_falls_back_to_the_first_two_leads():
    from eeg.mock import MotorRhythms

    m = MotorRhythms(["PO3", "PO4", "POz", "Pz"], FS, np.random.default_rng(0))
    assert m.mix[0, 0] == m.mix[1, 1] == 1.0 and m.mix.sum() == 2.0
    m.generate(int(FS), None)
    rest = m.generate(int(4 * FS), None)
    m.generate(int(FS), "right")
    right = m.generate(int(4 * FS), "right")
    assert right[:, 0].std() < 0.8 * rest[:, 0].std() and right[:, 1].std() > 0.9 * rest[:, 1].std()


def test_report_prints_the_trial_analysis(tmp_path):
    import scope

    rec, _ = _synthetic_trials()
    rec["meta"]["created"] = time.time()
    path = recording.save(rec, tmp_path, "cued")
    plain = recording.save(_recording(tmp_path)[1], tmp_path, "plain")
    text = scope.report([plain, path])
    assert text.count("Cued trials") == 1 and "Lateralisation C4 − C3" in text and "Classifier (left vs right" in text
    assert "device average" in text and "as recorded" in scope.report([path], "raw")


# -- cued trials: the window -------------------------------------------------------------------------
def test_window_runs_a_cued_session_and_shows_its_trials(tmp_path, monkeypatch):
    from rawscope.app import ScopeApp

    proto = cues.Protocol(per_class=1, practice=0, lead_in_s=0.3, baseline_s=0.3, cue_s=0.4, pause_s=(0.2, 0.3),
                          break_every=2, break_s=5.0, tail_s=0.2)
    feeds = Feeds(("flex", "headband"), config.EEG_DEVICES, mock=True, mock_motor=True, seed=0,
                  placement={"flex": {"O1": "C3", "O2": "C4", "T3": "Cz", "T4": "Fz"}})
    app = ScopeApp(feeds, tmp_path, protocol=proto)
    synth, _ = _synthetic_trials()
    synth["meta"]["created"] = time.time() - 60
    full = recording.save(synth, tmp_path, "synthetic")
    st = {"phase": 0}
    shots = {}

    def key(k):
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=k, mod=0, unicode="", scancode=0))

    def script(a, t):
        p = st["phase"]
        if a.cued is not None:
            shots.setdefault(a.cued.step.kind, a.cued.step.label)
            assert a.picker is None and a.feeds.recording
        if p == 0 and t > 0.4:
            key(pygame.K_t)
            st["phase"] = 1
        elif p == 1 and t > 0.5:
            assert a.prompt["kind"] == "cued"
            for ch in "sam":
                pygame.event.post(pygame.event.Event(pygame.TEXTINPUT, text=ch))
            key(pygame.K_RIGHT)                                       # imagine -> squeeze
            key(pygame.K_RETURN)
            st["phase"] = 2
        elif p == 2 and a.cued is not None and a.cued.step.kind == "break":
            a.open_picker("live", "flex", "O1")                       # can't move a lead mid-session
            assert a.picker is None
            key(pygame.K_SPACE)                                       # skip the break
            st["phase"] = 3
        elif p == 3 and a.cued is None:                               # finished: saved, analysis open
            assert a.view == "trials" and a.trial_view["path"].stem.endswith("sam")
            st["phase"] = 4
        elif p == 4:
            res = a.trial_result(wait=True)                           # windows follow the (short) protocol
            assert "error" not in res and res["n_trials"] == 3 and res["windows"]["task"] == (0.2, 0.4)
            assert res["classifier"]["flex"]["error"]                 # 1 trial per class: too few to classify
            a.draw()
            a.open_trials(full)
            res = a.trial_result(wait=True)
            for pos in [(120, 330), (1000, 300), (1300, 700), (1100, 520)]:   # hover: bars, curves, classifier
                monkeypatch.setattr(pygame.mouse, "get_pos", lambda pos=pos: pos)
                a.draw()
            monkeypatch.undo()
            key(pygame.K_RIGHT)
            key(pygame.K_a)                                           # as recorded
            st["phase"] = 5
        elif p == 5:
            assert a.trial_view["ref"] == "raw" and a.trial_view["focus"] == "Cz"
            assert a.trial_result(wait=True)["devices"]["flex"]["artifact"] > 1
            a.draw()
            key(pygame.K_p)                                           # play it back: cue ticks on the timeline
            st["phase"] = 6
        elif p == 6:
            assert a.view == "playback" and a.playback.trials
            a.seek(a.playback.trials[0]["cue"] - a.playback.t0 + 0.5)
            assert a.playback.cue_at(a.playback.head)[0] == 1
            a.draw()
            a.set_view("trials")
            key(pygame.K_ESCAPE)                                      # back to the list, not quit
            st["phase"] = 7
        elif p == 7:
            assert a.view == "compare" and a.running
            key(pygame.K_t)                                           # the focused recording's trials
            st["phase"] = 8
        elif p == 8:
            assert a.view == "trials"
            st["phase"] = 9
            a.running = False

    app.run(duration=12.0, script=script)
    assert st["phase"] == 9, st
    assert shots.get("cue") in cues.CLASSES and {"ready", "fixation", "pause", "break"} <= set(shots)
    meta = [m for m in recording.list_recordings(tmp_path) if m["name"] == "sam"][0]
    assert meta["protocol"]["task"] == "squeeze" and not meta["protocol"]["ended_early"]
    assert sorted(t["label"] for t in meta["trials"]) == sorted(cues.CLASSES)


def test_esc_ends_a_cued_session_early_and_keeps_the_finished_trials(tmp_path):
    from rawscope.app import ScopeApp

    proto = cues.Protocol(per_class=10, practice=0, lead_in_s=0.2, baseline_s=0.2, cue_s=0.3, pause_s=(0.1, 0.1))
    app = ScopeApp(Feeds(("flex",), config.EEG_DEVICES, mock=True, mock_motor=True, seed=0), tmp_path, protocol=proto)
    st = {}

    def script(a, t):
        if "go" not in st and t > 0.3:
            assert a.start_cued("imagine", name="early")
            st["go"] = t
        elif "esc" not in st and a.cued is not None and a.cued.completed >= 2 and a.cued.step.kind == "cue":
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=0, unicode="", scancode=0))
            st["esc"] = a.cued.completed
        elif "esc" in st and a.cued is None:
            st["running"] = a.running                                 # Esc ended the session, not the app
            a.running = False

    app.run(duration=5.0, script=script)
    meta = recording.list_recordings(tmp_path)[0]
    assert meta["name"] == "early" and meta["protocol"]["ended_early"] and st["running"]
    assert len(meta["trials"]) == st["esc"] == meta["protocol"]["completed"]
