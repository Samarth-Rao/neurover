import numpy as np
import pytest

from pipeline.features import Windower, extract, mean_absolute_value


def stream(windower, rate_hz, seconds, channels=8, t0=0.0, value=None):
    out = []
    for i in range(1, int(seconds * rate_hz) + 1):
        t = t0 + i / rate_hz
        x = np.full(channels, i if value is None else value, dtype=float)
        out += windower.push(t, x)
    return out


def test_emits_one_window_per_hop_after_the_first_full_window():
    w = Windower(0.2, 0.05)
    wins = stream(w, 50, 2.0)
    ends = [e for e, _ in wins]
    assert ends[0] == pytest.approx(0.02 + 0.2)          # first sample at 0.02 s
    assert np.allclose(np.diff(ends), 0.05)
    assert len(wins) == pytest.approx((2.0 - 0.22) / 0.05 + 1, abs=1)


def test_window_holds_exactly_the_last_window_s_of_samples():
    w = Windower(0.2, 0.05)
    for end, arr in stream(w, 50, 1.0):
        assert arr.shape == (10, 8)                        # 200 ms at 50 Hz
        times = arr[:, 0] / 50                             # sample i was pushed at t = i / 50
        assert np.all(times > end - 0.2 + 1e-9) and np.all(times <= end + 1e-9)
        assert np.allclose(np.diff(times), 0.02)


def test_works_at_other_rates_and_channel_counts():
    w = Windower(0.2, 0.05)
    wins = stream(w, 200, 1.0, channels=4)                 # e.g. raw Myo / a 4-channel EEG
    assert wins and all(a.shape == (40, 4) for _, a in wins)


def test_gap_in_stream_resyncs_instead_of_bursting_empty_windows():
    w = Windower(0.2, 0.05)
    stream(w, 50, 1.0)
    assert len(w.push(5.0, np.ones(8))) <= 1          # 4 s dropout: no burst of stale windows
    later = stream(w, 50, 0.5, t0=5.0)
    assert later and all(a.shape[0] >= 2 for _, a in later)


def test_mean_absolute_value_is_per_channel():
    window = np.array([[1.0, -2.0, 0.0], [-3.0, 2.0, 0.0]])
    assert np.allclose(mean_absolute_value(window), [2.0, 2.0, 0.0])
    assert np.allclose(extract(window, ("mav",)), [2.0, 2.0, 0.0])
