"""The command contract.

Every input source ultimately produces one of these five single-letter ASCII
commands, and every output sink consumes them. Nothing else crosses the
source -> sink boundary. The real micro:bit rover receives exactly these
letters over USB serial, one per line.
"""
from __future__ import annotations

from typing import Mapping, Optional

FORWARD = "F"
BACK = "B"
LEFT = "L"   # turn left in place
RIGHT = "R"  # turn right in place
STOP = "S"

ALL: tuple[str, ...] = (FORWARD, BACK, LEFT, RIGHT, STOP)

NAMES = {
    FORWARD: "FORWARD",
    BACK: "BACK",
    LEFT: "LEFT",
    RIGHT: "RIGHT",
    STOP: "STOP",
}


def is_command(value: object) -> bool:
    return isinstance(value, str) and value in ALL


def validate(value: object) -> str:
    if not is_command(value):
        raise ValueError(f"not a rover command: {value!r} (expected one of {', '.join(ALL)})")
    return value  # type: ignore[return-value]


def from_label(label: Optional[str], mapping: Mapping[str, str] | None = None) -> Optional[str]:
    """Map a source's raw label to a command.

    Direct sources (keyboard) already emit command letters. Signal sources emit
    class names (e.g. "fist") that are looked up in ``mapping``. ``None`` means
    the source has no opinion (e.g. no trained model yet) and maps to ``None``.
    """
    if label is None:
        return None
    if label in ALL:
        return label
    if mapping and label in mapping:
        return validate(mapping[label])
    return None


def encode(command: str) -> bytes:
    """Wire format for the real rover: the ASCII letter followed by a newline."""
    return (validate(command) + "\n").encode("ascii")
