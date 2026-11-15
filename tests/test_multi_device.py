"""Two EEG devices at once (BrainBit Flex + headband), no hardware: the stream merger, decoding
per device, a device dropping out and back, per-device calibration, bad electrodes left out."""
import json

import numpy as np
import pytest

import config
from pipeline import ssvep
from sources.eeg_merge import StreamMerger
from pipeline.controller import Controller
from sinks.sim_sink import SimSink
from sources.mock_eeg_source import MockEegSource

from test_ssvep import Clock, look

FS = 250.0


# -- merger --------------------------------------------------------------------------------
def _device_stream(t0, rate, seconds, freq, gap=None):
    t = t0 + np.arange(int(seconds * rate)) / rate
    if gap is not None:
        t = t[(t < gap[0]) | (t >= gap[1])]
    return t, np.sin(2 * np.pi * freq * t)[:, None] * np.array([[1.0, -2.0]])


def test_merger_aligns_two_clocks_and_marks_gaps():
    ta, xa = _device_stream(100.0, 250.0, 6.0, 10.0)
    tb, xb = _device_stream(100.0037, 250.05, 6.0, 7.0, gap=(102.0, 102.6))  # other crystal, a BLE gap
    m = StreamMerger([2, 2], FS, lag_s=0.3, max_gap_s=0.1)
    outs = []
    for now in np.arange(100.04, 106.5, 0.04):  # packets arrive every 40 ms
        for k, (t, x) in enumerate(((ta, xa), (tb, xb))):
            sel = (t <= now) & (t > now - 0.04)
            m.push(k, t[sel], x[sel])
        g, x, present = m.pull(now)
        assert len(g) == 0 or g[-1] <= now - 0.3 + 1e-9         # never ahead of real time minus the lag
        outs.append((g, x, present))
    g = np.concatenate([o[0] for o in outs])
    x = np.vstack([o[1] for o in outs])
    present = np.vstack([o[2] for o in outs])
    assert np.allclose(np.diff(g), 1 / FS)                      # one uniform 250 Hz timeline
    ok_a, ok_b = present[:, 0], present[:, 1]
    assert ok_a[1:].all()
    in_gap = (g > 102.0 + 0.1) & (g < 102.6 - 0.1)
    assert not ok_b[in_gap].any() and ok_b[(g > 103) & (g < 105)].all()
    # Interpolated values match each device's signal at the grid times.
    assert np.max(np.abs(x[ok_a, 0] - np.sin(2 * np.pi * 10 * g[ok_a]))) < 0.02
    assert np.max(np.abs(x[ok_b, 3] + 2 * np.sin(2 * np.pi * 7 * g[ok_b]))) < 0.03
    # During the gap the device's channels hold their last value (no step for the filters).
    held = x[in_gap, 2]
    assert np.ptp(held) == 0


def test_merger_produces_nothing_while_every_device_is_away_then_resumes():
    m = StreamMerger([1, 1], FS, lag_s=0.3)
    t1 = 10 + np.arange(250) / FS
    m.push(0, t1, np.ones((250, 1)))
    m.push(1, t1, np.ones((250, 1)))
    g, _, _ = m.pull(11.5)
    assert len(g) > 200
    assert len(m.pull(20.0)[0]) == 0          # both away: no data, so the dead-man stops the rover
    t2 = 30 + np.arange(250) / FS
    m.push(0, t2, np.ones((250, 1)))
    g, _, present = m.pull(31.5)
    assert g[0] >= 29.9 and present[:, 0].all() and not present[:, 1].any()   # skipped the empty stretch


# -- the EEG pipeline with two devices --------------------------------------------------------
class DropoutMock(MockEegSource):
    """Mock Flex + headband where the headband is away (held at zero, present=False) for a while."""

    def __init__(self, *a, away=None, **kw):
        super().__init__(*a, **kw)
        self.away = away
        hb = self.device_names.index("headband")
        self.hb_cols = [j for j, ch in enumerate(self.sdk_channels) if ch.startswith("headband:")]
        self.hb = hb

    def _read_chunk(self):
        t, x = super()._read_chunk()
        present = np.ones((len(t), len(self.device_names)), dtype=bool)
        if self.away is not None and len(t):
            gone = (t >= self.away[0]) & (t < self.away[1])
            present[gone, self.hb] = False
            x[np.ix_(gone, self.hb_cols)] = 0.0
        return t, x, present


def top(preds):
    """Which box scored highest in each window (the decoding itself, before any threshold)."""
    return [max(p.probs, key=p.probs.get) for p in preds]


def make2(tmp_path, cls=MockEegSource, seed=3, **kw):
    clock = Clock()
    src = cls(seed=seed, artifacts=False, clock=clock, models_dir=tmp_path,
              devices=("flex", "headband"), **kw)
    src.start()
    return src, clock


def test_both_devices_decode_and_combine(tmp_path):
    src, clock = make2(tmp_path)
    assert src.decoder.groups == ("flex", "headband")
    preds = look(src, clock, "left", 6.0)
    assert src.decoding == {"flex": ["PO3", "POz", "PO4"], "headband": ["O1", "O2"]}
    assert src.decoder.active == "flex+headband"
    tail = top(preds[-8:])
    assert tail.count("left") >= 7, tail


def test_device_dropout_decodes_on_the_other_and_rejoins_cleanly(tmp_path):
    src, clock = make2(tmp_path, cls=DropoutMock, away=(1004.0, 1009.0))
    look(src, clock, "up", 3.5)
    assert src.present_devices == ("flex", "headband")
    look(src, clock, "up", 3.0)                       # away from 1004.0: windows lose the headband
    assert src.present_devices == ("flex",) and src.decoder.active == "flex"
    assert all(src.quality[p] == "unknown" for p in src.device_positions["headband"])
    preds = look(src, clock, "up", 2.0)
    assert len(preds) >= 7 and top(preds).count("up") >= len(preds) - 1
    assert all(p.label in ("up", "rest") for p in preds)
    look(src, clock, "up", 4.0)                       # back at 1009.0 (+ a window length)
    assert src.present_devices == ("flex", "headband") and src.decoder.active == "flex+headband"
    assert src.artifact_windows == 0 and "headband" not in src.device_artifact   # filters restarted, no ringing
    look(src, clock, "up", 8.0)
    assert all(src.quality[p] == "good" for p in src.positions), src.quality


def test_noisy_electrode_is_left_out_of_decoding_not_vetoing_it(tmp_path):
    src, clock = make2(tmp_path)
    poz = src.positions.index("POz")
    orig = src.gen.generate

    def noisy(n):
        x = orig(n)
        x[:, poz] += src.gen.rng.normal(0, 150.0, n)  # badly seated: the Flex sits windows out at first
        return x

    src.gen.generate = noisy
    look(src, clock, "right", 6.0)
    assert src.quality["POz"] == "bad" and "NOISY" in src.quality_note["POz"]
    look(src, clock, "right", 3.0)
    assert src.decoding.get("flex") == ["PO3", "PO4"]          # decoded without POz
    preds = look(src, clock, "right", 3.0)
    assert len(preds) >= 10 and top(preds).count("right") >= len(preds) - 1
    st = src.status()
    assert not st.ok and "POz" in st.message                    # still told to fix it


def test_calibration_fits_every_device_combination_and_reloads(tmp_path):
    src, clock = make2(tmp_path, user="duo")
    for label in ["rest", *src.classes]:
        src.set_calibration_label(label)
        look(src, clock, None if label == "rest" else label, config.SSVEP_CAL_SEGMENT_S)
    cal = src.finish_calibration()
    assert cal.devices == "flex+headband"
    assert set(cal.by_devices) == {"flex+headband", "flex", "headband"}
    assert cal.setup == src.decode_setup() and "fused" in cal.setup
    assert cal.rest_recall >= config.SSVEP_CAL_MIN_REST_RECALL
    saved = json.loads(ssvep.calibration_path("duo", tmp_path).read_text())
    assert set(saved["by_devices"]) == set(cal.by_devices)

    again, clock2 = make2(tmp_path, cls=DropoutMock, user="duo", away=(1003.0, 1100.0))
    assert "loaded SSVEP calibration" in again.load_message
    look(again, clock2, None, 6.0)
    assert again.decoder.active == "flex"
    assert (again.decoder.threshold, again.decoder.margin) == tuple(cal.by_devices["flex"][:2])

    # A calibration made with other electrodes isn't applied.
    single, _ = make2(tmp_path, user="duo", **{})
    assert "loaded" in single.load_message
    other = MockEegSource(seed=1, artifacts=False, models_dir=tmp_path, user="duo", devices=("flex",))
    assert "different electrodes" in other.load_message
    assert other.decoder.threshold == config.SSVEP_THRESHOLD         # single-device default


# -- denoising ---------------------------------------------------------------------------------------
def _with_burst(src, pos, start_s, dur_s, make):
    """Add ``make(n)`` to electrode ``pos`` for dur_s, start_s seconds into the session."""
    i = src.positions.index(pos)
    orig = src.gen.generate
    a, b = int(start_s * src.fs), int((start_s + dur_s) * src.fs)

    def gen(n):
        x = orig(n)
        k0 = src.gen.n - n
        lo, hi = max(a, k0), min(b, k0 + n)
        if lo < hi:
            x[lo - k0:hi - k0, i] += make(hi - lo)
        return x

    src.gen.generate = gen


def test_short_electrode_pop_is_patched_not_dropped(tmp_path):
    src, clock = make2(tmp_path, ssvep_uv=6.0)
    _with_burst(src, "O1", 6.0, 0.15, lambda n: 600.0 * np.hanning(n))       # 0.15 s pop on the headband
    look(src, clock, "left", 5.5)
    seen_patched, before_drop = [], src.artifact_windows
    for _ in range(25):
        preds = look(src, clock, "left", 0.1)
        seen_patched.append(src.blanked)
        assert all(p.label in ("left", "rest") for p in preds)
    assert max(seen_patched) > 0                      # the pop was patched over...
    assert src.artifact_windows == before_drop        # ...not a single window thrown away
    assert "headband" in src.decoding


def test_muscle_burst_on_a_temple_electrode_is_blanked(tmp_path):
    src, clock = make2(tmp_path, ssvep_uv=6.0)
    rng = np.random.default_rng(0)
    _with_burst(src, "T3", 8.0, 0.3, lambda n: rng.normal(0, 25.0, n))       # jaw: 45-100 Hz-ish broadband
    look(src, clock, "right", 7.9)
    blanked = []
    for _ in range(15):
        preds = look(src, clock, "right", 0.1)
        blanked.append(src.blanked)
        assert all(p.label in ("right", "rest") for p in preds)   # never another direction
    assert max(blanked) > 0.05


def test_detrending_recovers_a_flicker_under_slow_drift():
    from pipeline.ssvep_cal import FeatureExtractor

    fs, n = 250.0, 500
    t = np.arange(n) / fs
    rng = np.random.default_rng(1)
    x = np.column_stack([np.sin(2 * np.pi * 12 * t) + rng.normal(0, 1.5, n) for _ in range(2)])
    x += np.column_stack([400 * t ** 2, -300 * t])          # drift a 6 Hz high-pass leaves behind
    fe = FeatureExtractor(fs, {"a": 12.0, "b": 15.0})
    plain = fe.window(x, {"d": [0, 1]}, need_z=False, detrend=0)["d"][0]
    clean = fe.window(x, {"d": [0, 1]}, need_z=False, detrend=2)["d"][0]
    assert clean[0] > plain[0] + 0.1 and clean[0] > clean[1]


def test_muscle_detector_is_the_same_live_and_offline():
    from pipeline.ssvep_cal import MuscleDetector

    rng = np.random.default_rng(2)
    x = rng.normal(0, 5.0, (5000, 3))
    x[2000:2200, 1] += rng.normal(0, 60.0, 200)               # a burst on one electrode
    whole = MuscleDetector(250.0, 3).process(x)
    det = MuscleDetector(250.0, 3)
    parts = np.vstack([det.process(x[i:i + 37]) for i in range(0, 5000, 37)])
    assert np.array_equal(whole, parts)
    assert whole[2050:2150, 1].all() and not whole[:1500].any()



# -- calibration that finds no flicker response, and the decoder view ---------------------------------------
def test_calibration_without_a_flicker_response_is_refused(tmp_path):
    """No response in the EEG: refuse (with the measured response) instead of installing 'never drive'."""
    import config as cfg
    from pipeline import ssvep_cal

    src, clock = make2(tmp_path, ssvep_uv=0.0)                 # the virtual driver's brain doesn't respond
    targets = src.decoder.targets
    segs = ssvep_cal.protocol(targets, blocks=3, trial_s=4.0, break_s=1.0, lead_s=1.0, seed=1)
    src.start_recording()
    for label, dur, _ in segs:
        src.set_calibration_label(label)
        look(src, clock, label if label in targets else None, dur)
    with pytest.raises(ValueError, match="no reliable flicker response"):
        src.finish_calibration()
    assert not ssvep_cal.Path(tmp_path / "ssvep_default.json").exists()
    assert src.decoder.model is None                            # still the uncalibrated decoder


def test_decoder_view_shows_which_box_stands_out(tmp_path):
    src, clock = make2(tmp_path, ssvep_uv=6.0)
    look(src, clock, "left", 5.0)
    for dev in ("flex", "headband"):
        resp = src.response[dev]
        assert resp["left"] > 2.0 and resp["left"] == max(resp.values())
    assert len(src.score_history) > 20 and src.score_history[-1][2] == "left"


def test_decoder_view_draws_for_eeg_and_emg(tmp_path):
    import pygame

    from sources.mock_emg_source import MockEmgSource
    from ui.app import App

    for src in (MockEegSource(seed=1, artifacts=False, models_dir=tmp_path),
                MockEmgSource(seed=1, models_dir=tmp_path, autoload=False)):
        sim = SimSink()
        ctl = Controller(src, [sim])
        app = App(src, ctl, [sim], sim=sim)
        done = set()

        def held(t):
            if "i" not in done:
                done.add("i")
                pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_i, mod=0))
            return {"up"}

        src.start()
        app.run(duration=2.5, held_provider=held)
        assert app.show_decoding
        src.stop()
