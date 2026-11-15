"""BrainBit Flex source against a fake pyneurosdk2 (tests/fake_neurosdk), no hardware."""
import sys
import time
from pathlib import Path

import numpy as np
import pytest

import commands as C
import config
from pipeline.controller import Controller
from sinks.sim_sink import SimSink
from sources.eeg_source import BrainBitSource

FAKE = str(Path(__file__).parent / "fake_neurosdk")
# One device, any BrainBit family, electrodes where the SDK names say (the original single-Flex setup).
ONE = {"bb": {"label": "BrainBit", "family": "any", "map": {"O1": "O1", "O2": "O2", "T3": "P3", "T4": "P4"}}}
# config.EEG_DEVICES without the serial pins (the fake devices have other serials)
UNPINNED = {n: {k: v for k, v in d.items() if k != "serial"} for n, d in config.EEG_DEVICES.items()}


@pytest.fixture
def brainbit(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(FAKE)
    for mod in [m for m in sys.modules if m == "neurosdk" or m.startswith("neurosdk.")]:
        monkeypatch.delitem(sys.modules, mod)
    monkeypatch.setattr(config, "BRAINBIT_RESIST_S", 0.4)
    monkeypatch.setattr(config, "BRAINBIT_RECONNECT_S", 0.3)
    import sources.brainbit_worker as bw
    monkeypatch.setattr(bw, "_CLAIMS", {})      # which device each link has: fresh per test
    monkeypatch.setattr(bw, "_KIND_OF", {})
    made = []

    def make(scenario, **kw):
        monkeypatch.setenv("FAKE_BRAINBIT", scenario)
        kw.setdefault("devices", ONE)
        src = BrainBitSource(models_dir=tmp_path, **kw)
        made.append(src)
        return src

    yield make
    for src in made:
        src.stop()


def pump(src, seconds, until=None, ctl=None):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        ctl.tick() if ctl is not None else src.poll()
        if until is not None and until():
            return True
        time.sleep(0.01)
    return until is None


def test_flex4_streams_microvolts_mapped_by_channel_name(brainbit):
    src = brainbit("flex4")
    src.start()
    assert pump(src, 10, until=lambda: src.state == "streaming" and src.clean_windows > 2), src.status()
    assert "Flex" in src.device_info["kind"]
    st = src.status()
    assert st.ok and "battery 87%" in st.message
    # The fake reports channels as T3, O1, T4, O2 (Num 0..3); O1's 40 µV marker must land on "O1".
    amp = src.traces()[-250:].std(axis=0)
    assert src.positions[int(np.argmax(amp))] == "O1"
    assert 20 < amp.max() < 35          # 40 µV sine -> ~28 µV RMS: volts were converted to µV
    assert all(src.contact[p] for p in src.positions)
    assert all(2e5 < r < 4e5 for r in src.resistance.values())


def test_poor_contact_is_reported_but_streaming_continues(brainbit):
    src = brainbit("badcontact")
    src.start()
    assert pump(src, 10, until=lambda: src.state == "streaming" and src.clean_windows > 2)
    pz = src.position_of("T4")          # the fake's bad electrode is SDK channel T4
    assert src.contact[pz] is False
    assert pump(src, 5, until=lambda: src.quality[pz] == "bad")   # the latched electrode check agrees
    st = src.status()
    assert not st.ok and "HIGH RESISTANCE" in st.message and pz in st.message


def test_no_headband_found_keeps_scanning_without_crashing(brainbit):
    src = brainbit("none")
    src.start()
    assert pump(src, 3, until=lambda: src.state == "scanning")
    st = src.status()
    assert not st.ok and st.pending and "scanning" in st.message
    t0 = time.monotonic()
    src.stop()
    assert time.monotonic() - t0 < 3.0


def test_flex_v1_family_is_supported(brainbit):
    src = brainbit("flex1")
    src.start()
    assert pump(src, 10, until=lambda: src.state == "streaming" and src.clean_windows > 2)
    assert src.device_info["kind"] == "BrainBit Flex (v1)"
    assert src.positions[int(np.argmax(src.traces()[-250:].std(axis=0)))] == "O1"
    assert all(src.contact[p] for p in src.positions)


def test_bluetooth_drop_forces_stop_then_reconnects(brainbit):
    src = brainbit("drop")
    ctl = Controller(src, [SimSink()])
    src.start()
    assert pump(src, 10, ctl=ctl, until=lambda: src.state == "streaming" and src.clean_windows > 0)
    assert pump(src, 6, ctl=ctl, until=lambda: src.state == "reconnecting")
    assert pump(src, 3, ctl=ctl, until=lambda: ctl.smoother.deadman)
    assert ctl.command == C.STOP
    assert pump(src, 10, ctl=ctl, until=lambda: src.state == "streaming")
    before = src.clean_windows
    assert pump(src, 6, ctl=ctl, until=lambda: src.clean_windows > before + 2)  # data flows again


def test_r_remeasures_contact(brainbit):
    src = brainbit("flex4")
    assert not src.measure_resistance()   # not streaming yet
    src.start()
    assert pump(src, 10, until=lambda: src.state == "streaming")
    assert src.measure_resistance()
    assert pump(src, 5, until=lambda: src.state == "resist")
    assert pump(src, 5, until=lambda: src.state == "streaming")


def test_missing_sdk_is_an_on_screen_error(brainbit, monkeypatch):
    import sources.eeg_source as es

    monkeypatch.setattr(es.importlib.util, "find_spec", lambda name: None)
    src = brainbit("flex4")
    src.start()
    st = src.status()
    assert not st.ok and "pip install pyneurosdk2" in st.message


def test_alpha_mode_works_with_the_brainbit_source(brainbit):
    src = brainbit("flex4", mode="alpha")
    src.start()
    assert pump(src, 10, until=lambda: src.state == "streaming" and src.clean_windows > 2)
    assert not src.supports_stimulus and src.latest() is not None


# -- Flex + headband together --------------------------------------------------------------
def test_flex_and_headband_stream_together_on_one_timeline(brainbit):
    src = brainbit("dual", devices=UNPINNED)     # config.EEG_DEVICES maps
    src.start()
    assert pump(src, 15, until=lambda: src.state == "streaming" and src.clean_windows > 4
                and set(src.present_devices) == {"flex", "headband"}), src.status()
    st = src.status()
    assert st.ok and "Flex + Headband streaming" in st.message and "Headband 64%" in st.message
    assert len(src.positions) == 8 and {"PO3", "POz", "PO4", "Pz", "O1", "O2", "T3", "T4"} == set(src.positions)
    amp = dict(zip(src.positions, src.traces()[-250:].std(axis=0)))
    # Flex lead "O1" (40 µV marker) is plugged in at PO3; the headband's O2 (30 µV marker) is O2.
    assert 20 < amp["PO3"] < 35 and 14 < amp["O2"] < 26
    assert all(amp[p] < 6 for p in amp if p not in ("PO3", "O2"))
    # Both devices' electrodes are decoded, each device separately (scores averaged).
    assert src.decoding == {"flex": ["PO3", "POz", "PO4"], "headband": ["O1", "O2"]}
    assert src.decoder.active == "flex+headband"
    # Two devices: no contact measurement while connecting (it stalls a real Flex v1)...
    assert all(src.contact[p] is None for p in src.positions)
    # ...but R measures them one at a time, then both stream again.
    assert src.measure_resistance()
    both_in_resist = []
    def done():
        both_in_resist.append(sum(d.state == "resist" for d in src.links))
        return set(src.resistance) == set(src.positions) and all(d.state == "streaming" for d in src.links)
    assert pump(src, 25, until=done), src.status()
    assert all(src.contact[p] for p in src.positions)
    before = src.clean_windows
    assert pump(src, 6, until=lambda: src.clean_windows > before + 3
                and set(src.present_devices) == {"flex", "headband"})


def test_headband_drop_keeps_driving_on_the_flex_then_rejoins(brainbit):
    src = brainbit("dual-drop", devices=UNPINNED)
    ctl = Controller(src, [SimSink()])
    src.start()
    assert pump(src, 15, ctl=ctl, until=lambda: set(src.present_devices) == {"flex", "headband"})
    assert pump(src, 8, ctl=ctl, until=lambda: src.links[1].state == "reconnecting")
    # The Flex alone keeps producing decoded windows while the headband is away.
    assert pump(src, 3, ctl=ctl, until=lambda: src.present_devices == ("flex",))
    before = src.clean_windows
    assert pump(src, 3, ctl=ctl, until=lambda: src.clean_windows > before + 3)
    assert src.decoder.active == "flex" and "headband" not in src.decoding
    st = src.status()
    assert not st.ok and st.pending and "decoding without it" in st.message
    assert not ctl.smoother.deadman
    # Headband electrodes aren't judged while it's away (no false FLAT warnings).
    assert all(src.quality[p] != "bad" for p in ("O1", "O2", "T3", "T4"))
    # Then it reconnects and is used again, without a filter-ringing artifact on its channels.
    assert pump(src, 15, ctl=ctl, until=lambda: set(src.present_devices) == {"flex", "headband"})
    assert pump(src, 4, ctl=ctl, until=lambda: "headband" in src.decoding)
    assert all(src.quality[p] != "bad" for p in src.positions)


def test_two_devices_start_one_at_a_time(brainbit):
    """The second device connects only once the first is delivering samples."""
    src = brainbit("dual", devices=UNPINNED)
    src.start()
    order = []
    def watch():
        for d in src.links:
            if d.state in ("connecting", "streaming") and (d.name, d.state) not in order:
                order.append((d.name, d.state))
        return all(d.state == "streaming" for d in src.links)
    assert pump(src, 15, until=watch)
    first = order[0][0]
    other = next(n for n, _ in order if n != first)
    assert order.index((first, "streaming")) < order.index((other, "streaming"))
    assert not any(d.state == "resist" for d in src.links)


@pytest.mark.parametrize("pinned", [True, False])
def test_flex_v1_and_headband_look_alike_but_land_in_the_right_slots(brainbit, monkeypatch, pinned):
    """Both advertise as LEBrainBit "BrainBit"; only the firmware tells a Flex v1 from the headband.
    Unpinned, the Flex slot meets the (stronger) headband first, hands it over, and finds the Flex."""
    devices = {n: dict(d) for n, d in config.EEG_DEVICES.items()} if pinned else UNPINNED
    src = brainbit("dual-v1", devices=devices)
    src.start()
    assert pump(src, 20, until=lambda: set(src.present_devices) == {"flex", "headband"}
                and src.clean_windows > 3), src.status()
    kinds = {d.name: d.info.get("kind") for d in src.links}
    assert kinds == {"flex": "BrainBit Flex (v1)", "headband": "BrainBit headband"}
    amp = dict(zip(src.positions, src.traces()[-250:].std(axis=0)))
    assert 20 < amp["PO3"] < 35 and 14 < amp["O2"] < 26      # Flex O1 marker at PO3, headband O2 marker at O2
