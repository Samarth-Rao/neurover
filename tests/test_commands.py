import pytest

import commands as C
import config


def test_contract_is_exactly_five_ascii_letters():
    assert C.ALL == ("F", "B", "L", "R", "S")
    assert all(len(c) == 1 and c.isascii() for c in C.ALL)


def test_gesture_to_command_map_matches_spec():
    assert config.GESTURE_TO_COMMAND == {
        "rest": "S", "fist": "F", "spread": "B", "wave_in": "L", "wave_out": "R",
    }


def test_every_gesture_maps_to_a_valid_command_and_has_a_training_key():
    assert set(config.GESTURE_TO_COMMAND) == set(config.GESTURES)
    assert all(C.is_command(c) for c in config.GESTURE_TO_COMMAND.values())
    assert config.GESTURE_KEYS == {0: "rest", 1: "fist", 2: "spread", 3: "wave_in", 4: "wave_out"}


def test_from_label():
    m = config.GESTURE_TO_COMMAND
    assert C.from_label("F", m) == "F"          # direct sources emit letters
    assert C.from_label("fist", m) == "F"
    assert C.from_label("wave_in", m) == "L"
    assert C.from_label(None, m) is None
    assert C.from_label("unknown", m) is None


def test_wire_encoding():
    assert C.encode("F") == b"F\n"
    assert C.encode("S") == b"S\n"
    with pytest.raises(ValueError):
        C.encode("forward")
