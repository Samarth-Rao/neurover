import pytest

import commands as C
import config


def test_contract_is_exactly_five_ascii_letters():
    assert C.ALL == ("F", "B", "L", "R", "S")
    assert all(len(c) == 1 and c.isascii() for c in C.ALL)


def test_every_flicker_box_maps_to_a_command_and_the_centre_stops():
    assert config.SSVEP_COMMANDS == {"up": "F", "left": "L", "right": "R", "rest": "S"}
    assert set(config.SSVEP_FRAMES) | {"rest"} == set(config.SSVEP_COMMANDS)


def test_from_label():
    m = config.SSVEP_COMMANDS
    assert C.from_label("F", m) == "F"          # direct sources (keyboard) emit letters
    assert C.from_label("left", m) == "L"
    assert C.from_label("rest", m) == "S"
    assert C.from_label(None, m) is None
    assert C.from_label("unknown", m) is None


def test_wire_encoding():
    assert C.encode("F") == b"F\n"
    assert C.encode("S") == b"S\n"
    with pytest.raises(ValueError):
        C.encode("forward")
