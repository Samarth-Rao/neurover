"""The clickable head map for saying where an electrode is. Moving one and remembering where they all
are (scope_placement.json, which the rover uses too) is in eeg/devices.py."""
from __future__ import annotations

import math

from eeg.devices import PLACEMENT_PATH, reassign  # noqa: F401  (re-exported for the scope)
from eeg.devices import load_placement as load  # noqa: F401
from eeg.devices import save_placement as save  # noqa: F401


def _ring(theta_deg: float, r: float = 1.0) -> tuple[float, float]:
    """Head-map coordinates: x to the right ear, y to the nose; the circle (r = 1) is the 10% line
    through Fpz, T3, Oz and T4. ``theta`` is degrees clockwise from the nose."""
    t = math.radians(theta_deg)
    return (r * math.sin(t), r * math.cos(t))


# 10-20 positions (BrainBit's old names T3/T4/T5/T6), the parieto-occipital row (10-10, plus PO1/PO2 from 10-5) and Iz.
HEAD: dict[str, tuple[float, float]] = {
    "Fp1": _ring(-18), "Fpz": _ring(0), "Fp2": _ring(18),
    "F7": _ring(-54), "F3": (-0.40, 0.50), "Fz": (0.0, 0.50), "F4": (0.40, 0.50), "F8": _ring(54),
    "T3": _ring(-90), "C3": (-0.50, 0.0), "Cz": (0.0, 0.0), "C4": (0.50, 0.0), "T4": _ring(90),
    "T5": _ring(-126), "P3": (-0.40, -0.50), "Pz": (0.0, -0.50), "P4": (0.40, -0.50), "T6": _ring(126),
    "PO7": _ring(-144), "PO3": (-0.36, -0.74), "PO1": (-0.18, -0.70), "POz": (0.0, -0.74), "PO2": (0.18, -0.70),
    "PO4": (0.36, -0.74), "PO8": _ring(144),
    "O1": _ring(-162), "Oz": _ring(180), "O2": _ring(162),
    "Iz": (0.0, -1.18),
}
