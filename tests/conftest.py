import os
import sys
from pathlib import Path

# Headless pygame for UI smoke tests.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Tests use a fixed layout (the Flex on the parieto-occipital row, the headband on), whatever config.py
# says for today's session. Set before any app module imports config's values as defaults.
import config  # noqa: E402

config.EEG_DEVICES = {
    "flex": {"label": "Flex", "family": "flex", "serial": "450904",
             "map": {"O1": "PO3", "O2": "PO4", "T3": "POz", "T4": "Pz"}},
    "headband": {"label": "Headband", "family": "headband", "serial": "133013",
                 "map": {"O1": "O1", "O2": "O2", "T3": "T3", "T4": "T4"}},
}
config.EEG_USE_DEVICES = ("flex", "headband")
