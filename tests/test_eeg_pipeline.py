"""EEG plumbing: filters, artifact rejection, channel mapping, stimulus timing, warning gate."""
import numpy as np
import pytest

import commands as C
import config
from pipeline.controller import Controller
from pipeline.eeg import StreamFilter, channel_permutation
from sinks.sim_sink import SimSink
from sources.eeg_common import EegStreamSource
from sources.mock_eeg_source import MockEegSource
from ui.stimulus import CalibrationRun, Stimulus, StimulusLayout, frame_pattern

FPS = 60


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


# -- filters -----------------------------------------------------------------------------
def test_stream_filter_removes_dc_offset_and_mains():
    fs = 250.0
    t = np.arange(int(10 * fs)) / fs
    x = (300.0 + 20 * np.sin(2 * np.pi * 60 * t) + 5 * np.sin(2 * np.pi * 12 * t))[:, None]
    flt = StreamFilter(fs, config.SSVEP_BAND, notch_hz=60.0)
    y = np.vstack([flt.process(x[i:i + 7]) for i in range(0, len(x), 7)])  # chunked like the live stream
    tail = y[int(5 * fs):, 0]
    assert abs(tail.mean()) < 1.0                  # DC gone
    assert 4.0 < tail.std() * np.sqrt(2) < 6.0     # the 12 Hz signal survives, mains doesn't


# -- artifacts ---------------------------------------------------------------------------
def test_artifact_windows_are_dropped_and_never_drive(tmp_path):
    clock = Clock()
    src = MockEegSource(seed=4, artifacts=False, clock=clock, models_dir=tmp_path)
    src.gen.ssvep_uv = 4.0  # a clear response, so the rover is definitely driving before the clench
    src.start()
    sim = SimSink()
    ctl = Controller(src, [sim], clock=clock)

    def run(held, seconds):
        src.on_input(frozenset(held))
        seen = []
        for _ in range(round(seconds * FPS)):
            clock.t += 1.0 / FPS
            ctl.tick(clock.t)
            seen.append((ctl.command, src.artifact))
        return seen

    run({"up"}, 8.0)
    assert ctl.command == C.FORWARD
    before = src.clean_windows
    seen = run({"up", "jaw"}, 3.0)                 # clench while still looking at the F box
    assert src.artifact_windows > 0
    # Only windows that are still mostly from before the clench get through: up to SSVEP_MAX_BLANKED of
    # a window (incl. padding) is patched over and the clean rest decoded; beyond that it's dropped.
    patched = round((config.SSVEP_MAX_BLANKED * src.window_s - config.EEG_BLANK_PAD_S) / src.hop_s) + 2
    assert src.clean_windows - before <= patched
    assert ctl.command == C.STOP and ctl.smoother.deadman
    # Once the artifact is detected the rover is stopped within the dead-man time, and stays stopped.
    first = next(i for i, (_, art) in enumerate(seen) if art)
    after = [cmd for cmd, _ in seen[first + round((config.DEADMAN_S + 0.1) * FPS):]]
    assert after and all(cmd == C.STOP for cmd in after)

    # An artifact alone (eyes on the centre) never becomes a drive command.
    seen = run({"jaw"}, 3.0)
    assert all(cmd == C.STOP for cmd, _ in seen)


# -- channel mapping ---------------------------------------------------------------------
class InjectedEeg(EegStreamSource):
    """EEG source fed with hand-made SDK-ordered chunks."""
    name = "injected"

    def __init__(self, chunks, **kw):
        super().__init__(models_dir=kw.pop("models_dir"), **kw)
        self._chunks = list(chunks)

    def start(self):
        pass

    def stop(self):
        pass

    def _read_chunk(self):
        return self._chunks.pop(0) if self._chunks else (np.zeros(0), np.zeros((0, len(self.sdk_channels))))


def _marker_chunks(marker_channel: int, seconds=6.0, fs=250.0):
    n = int(seconds * fs)
    t = 1000.0 + np.arange(1, n + 1) / fs
    x = np.random.default_rng(0).normal(0, 2.0, (n, 4))
    x[:, marker_channel] += 40 * np.sin(2 * np.pi * 5 * t)  # big 5 Hz marker on one SDK channel
    return [(t[i:i + 50], x[i:i + 50]) for i in range(0, n, 50)]


@pytest.mark.parametrize("mapping,expect", [
    ({"O1": "O1", "O2": "O2", "T3": "Oz", "T4": "Pz"}, "Pz"),
    ({"O1": "Pz", "O2": "O1", "T3": "O2", "T4": "Oz"}, "Oz"),
])
def test_channel_mapping_config_is_respected(tmp_path, mapping, expect):
    src = InjectedEeg(_marker_chunks(marker_channel=3), channel_map=mapping, models_dir=tmp_path,
                      sdk_channels=("O1", "O2", "T3", "T4"))
    assert src.positions == ("O1", "Oz", "O2", "Pz")           # always shown in head order
    for _ in range(40):
        src.poll()
    amp = src.traces()[-500:].std(axis=0)
    assert src.positions[int(np.argmax(amp))] == expect          # SDK "T4" shows up where it was placed
    occ = {src.positions[i] for i in src.occipital}
    assert occ == {"O1", "Oz", "O2"}


def test_bad_channel_maps_are_rejected():
    with pytest.raises(ValueError):
        channel_permutation(("O1", "O2"), {"O1": "O1"}, ("O1",))
    with pytest.raises(ValueError):
        channel_permutation(("O1", "O2"), {"O1": "Oz", "O2": "Oz"}, ("Oz",))


# -- stimulus ------------------------------------------------------------------------------
@pytest.mark.parametrize("frames,hz", [(7, 8.57), (6, 10.0), (5, 12.0), (4, 15.0)])
def test_stimulus_frame_patterns_have_the_right_cycle_length(frames, hz):
    pat = frame_pattern(frames)
    assert len(pat) == frames
    assert sum(pat) == (frames + 1) // 2
    assert sum(1 for a, b in zip(pat, pat[1:] + pat[:1]) if a and not b) == 1  # one on->off per cycle
    stim = Stimulus({"x": frames})
    seq = []
    for _ in range(frames * 10):
        seq.append(stim.is_on("x"))
        stim.advance()
    assert seq == pat * 10                                       # repeats every `frames` frames
    assert stim.frequencies(60.0)["x"] == pytest.approx(hz, abs=0.01)


def test_stimulus_layout_boxes_are_big_and_far_apart():
    import math

    import pygame

    lay = StimulusLayout.fit(pygame.Rect(0, 0, 1600, 900), 220, 16)
    boxes = list(lay.boxes.values())
    assert all(b.w >= 200 and b.h >= 200 for b in boxes)
    c = {k: v.center for k, v in lay.boxes.items()}
    diagonal = [math.dist(c[a], c[b]) for a in ("up", "down") for b in ("left", "right")]
    assert min(diagonal) > 700                    # was ~477 px when the boxes surrounded the arena only
    assert math.dist(c["up"], c["down"]) > 600    # limited by the screen height (as before)
    others = [lay.centre, *lay.panels.values()]
    for b in boxes:
        assert not any(b.colliderect(o) for o in others)
    for i, a in enumerate(others):
        assert not any(a.colliderect(o) for o in others[i + 1:])
    assert lay.centre.w > 900 and lay.centre.h > 250
    assert all(p.w > 500 and p.h > 200 for p in lay.panels.values())


def test_calibration_run_sequence():
    run = CalibrationRun([("rest", 20.0, "centre"), ("up", 20.0, "up"), ("down", 20.0, "down")], start=0.0)
    assert [s for s, _, _ in run.segments] == ["rest", "up", "down"]
    assert run.current(5.0)[1] == "rest"
    assert run.current(25.0)[:2] == (1, "up")
    assert run.current(59.0)[1] == "down"
    assert run.current(60.5) is None


def test_photosensitivity_warning_gates_the_flicker(tmp_path):
    import pygame

    from ui.app import App

    src = MockEegSource(seed=1, models_dir=tmp_path)
    sim = SimSink()
    ctl = Controller(src, [sim])
    app = App(src, ctl, [sim], sim=sim)
    seen = {}
    plan = [(0.1, pygame.K_v), (0.3, pygame.K_ESCAPE), (0.5, pygame.K_v), (0.7, pygame.K_RETURN),
            (0.9, pygame.K_v), (1.1, pygame.K_v)]
    checks = {0.2: "warning_after_v", 0.4: "after_escape", 0.8: "after_enter", 1.0: "after_v_off", 1.2: "after_v_on"}
    done = set()

    def held(t):
        for at, key in plan:
            if t >= at and at not in done:
                done.add(at)
                pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key, mod=0))
        for at, name in checks.items():
            if t >= at and name not in seen:
                seen[name] = (app.warning_open, app.stimulus.enabled, app.running)
        return set()

    src.start()
    assert app.stimulus is not None and not app.stimulus.enabled      # off by default
    app.run(duration=1.4, held_provider=held)
    assert seen["warning_after_v"] == (True, False, True)              # V shows the warning, no flicker yet
    assert seen["after_escape"] == (False, False, True)                # Esc cancels, app keeps running
    assert seen["after_enter"] == (False, True, True)                  # Enter accepts -> flicker on
    assert seen["after_v_off"] == (False, False, True)
    assert seen["after_v_on"] == (False, True, True)                   # accepted once per session


def test_calibration_saves_a_raw_recording(tmp_path):
    """A full guided calibration through the app saves EEG + segments + flicker frame times."""
    import pygame

    import config as cfg
    from pipeline import ssvep_data
    from ui.app import App

    src = MockEegSource(seed=2, models_dir=tmp_path, user="tester")
    sim = SimSink()
    ctl = Controller(src, [sim])
    app = App(src, ctl, [sim], sim=sim)
    segs = [("ready", 0.4, "get ready"), ("rest", 1.0, "centre"), ("up", 1.0, "up"), ("down", 1.0, "down"),
            ("break", 0.4, "blink"), ("left", 1.0, "left"), ("right", 1.0, "right"), ("rest", 1.0, "centre")]
    src.calibration_segments = lambda: segs   # a short protocol for the test
    done = set()

    def held(t):
        for at, key in ((0.1, pygame.K_c), (0.3, pygame.K_RETURN)):      # C -> warning -> Enter
            if t >= at and at not in done:
                done.add(at)
                pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key, mod=0))
        return set()

    src.start()
    app.run(duration=0.5 + sum(d for _, d, _ in segs) + 1.5, held_provider=held)
    files = ssvep_data.list_recordings(tmp_path, "ssvep", "tester")
    assert len(files) == 1
    rec = ssvep_data.load_recording(files[0])
    assert [s[0] for s in rec["segments"]] == [s[0] for s in segs]     # as they happened, incl. breaks
    # The fit ran in the background (far too little data here: it fails cleanly) and the rover resumed.
    assert app.fitting is None and not ctl.paused and app.calibration is None
    assert rec["x"].shape[1] == len(src.positions) and rec["positions"] == list(src.positions)
    assert rec["devices"] == {n: list(p) for n, p in src.device_positions.items()}
    assert rec["present"].shape == (len(rec["t"]), len(src.device_names)) and rec["present"].all()
    span = rec["t"][-1] - rec["t"][0]
    assert abs(len(rec["t"]) / span - cfg.EEG_SAMPLE_HZ) < 10
    assert len(rec["flip_t"]) > 100 and np.all(np.diff(rec["flip_frame"]) >= 1)
    assert rec["frames"] == cfg.SSVEP_FRAMES


# -- electrode quality ---------------------------------------------------------------------
def _chunks_with_noisy_channel(seconds, noisy_channel=None, t0=1000.0, fs=250.0, noisy_uv=300.0, seed=0):
    n = int(seconds * fs)
    t = t0 + np.arange(1, n + 1) / fs
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 8.0, (n, 4)) + 200.0                 # ~8 µV EEG + electrode DC offset
    if noisy_channel is not None:
        x[:, noisy_channel] += rng.normal(0, noisy_uv, n)    # badly seated electrode
    return [(t[i:i + 50], x[i:i + 50]) for i in range(0, n, 50)]


def test_bad_electrode_is_flagged_latched_and_blocks_nothing_else(tmp_path):
    good = _chunks_with_noisy_channel(6.0)
    bad = _chunks_with_noisy_channel(4.0, noisy_channel=3, t0=1006.0, seed=1)   # SDK "T4" -> P4
    recover = _chunks_with_noisy_channel(16.0, t0=1010.0, seed=2)
    src = InjectedEeg(good + bad + recover, models_dir=tmp_path, sdk_channels=("O1", "O2", "T3", "T4"),
                      channel_map={"O1": "O1", "O2": "O2", "T3": "P3", "T4": "P4"})
    p4 = src.position_of("T4")
    seen = []
    for _ in range(len(good) + len(bad) + len(recover)):
        src.poll()
        seen.append((src.quality[p4], src._amp_hist[-1][0] if src._amp_hist else 0.0))
    states = [s for s, _ in seen]
    assert "bad" in states
    first_bad = states.index("bad")
    # Latched: stays bad well after the noise stopped (recovery needs EEG_QUALITY_RECOVER_S of good signal).
    noise_end = 1010.0
    bad_after_noise = [s for s, t in seen if noise_end < t < noise_end + 4.0]
    assert bad_after_noise and all(s == "bad" for s in bad_after_noise)
    assert states[-1] == "good"                    # ~10 s after reseating: 6 s median clears + 5 s latch
    assert first_bad > 0 and states[0] != "bad"
    others = [p for p in src.positions if p != p4]
    assert all(src.quality[p] == "good" for p in others)


def test_noisy_decode_electrode_turns_status_red_and_blocks_calibration(tmp_path):
    import pygame

    from ui.app import App

    src = MockEegSource(seed=1, artifacts=False, models_dir=tmp_path)
    o1 = src.positions.index("O1")
    orig = src.gen.generate

    def noisy(n):
        x = orig(n)
        x[:, o1] += src.gen.rng.normal(0, 300.0, n)          # O1 (used for decoding) badly seated
        return x

    src.gen.generate = noisy
    sim = SimSink()
    ctl = Controller(src, [sim])
    app = App(src, ctl, [sim], sim=sim)
    seen = {}

    def held(t):
        if t >= 4.0 and "c" not in seen:
            seen["c"] = True
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_c, mod=0))
        return set()

    src.start()
    app.run(duration=4.6, held_provider=held)
    st = src.status()
    assert src.quality["O1"] == "bad" and "NOISY" in src.quality_note["O1"]
    assert not st.ok and not st.pending and "BAD ELECTRODE: O1" in st.message   # red, names the electrode
    assert app.calibration is None and not app.warning_open                       # C refused



def test_slightly_noisy_electrode_is_marginal_not_bad(tmp_path):
    """80-150 µV: a yellow note, but still decoded and calibration isn't blocked."""
    chunks = _chunks_with_noisy_channel(10.0, noisy_channel=3, noisy_uv=65.0, seed=4)   # ~100 µV peaks
    src = InjectedEeg(chunks, models_dir=tmp_path, sdk_channels=("O1", "O2", "T3", "T4"),
                      channel_map={"O1": "O1", "O2": "O2", "T3": "Oz", "T4": "P4"})
    for _ in range(len(chunks)):
        src.poll()
    p4 = src.position_of("T4")
    assert src.quality[p4] == "marginal" and "MARGINAL" in src.quality_note[p4]
    assert not src.bad_electrodes() and src.electrode_warning() is None



def test_slow_settling_drift_does_not_count_as_noise_for_ssvep(tmp_path):
    """Fresh dry electrodes drift by thousands of µV at 1-3 Hz; SSVEP decodes 6-40 Hz, so that's fine."""
    n, fs = int(10 * 250), 250.0
    t = 1000.0 + np.arange(1, n + 1) / fs
    rng = np.random.default_rng(6)
    x = rng.normal(0, 8.0, (n, 4)) + 200.0
    x[:, 0] += 1500 * np.sin(2 * np.pi * 1.5 * t) + 800 * np.sin(2 * np.pi * 2.7 * t + 1)   # settling drift
    chunks = [(t[i:i + 50], x[i:i + 50]) for i in range(0, n, 50)]
    src = InjectedEeg(chunks, models_dir=tmp_path, sdk_channels=("O1", "O2", "T3", "T4"),
                      channel_map={"O1": "O1", "O2": "O2", "T3": "Oz", "T4": "P4"})
    for _ in range(len(chunks)):
        src.poll()
    assert src.quality["O1"] in ("good", "marginal") and "O1" not in src.bad_electrodes()
