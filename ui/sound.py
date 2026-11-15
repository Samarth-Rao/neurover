"""Short beeps for guided calibration (you can't read the screen with your eyes closed)."""
from __future__ import annotations

import numpy as np
import pygame

_ready: bool | None = None


def beep(freq: float = 880.0, ms: int = 160, volume: float = 0.35) -> None:
    """Play a short tone. Silently does nothing if audio isn't available."""
    global _ready
    try:
        if _ready is None:
            try:
                pygame.mixer.init(frequency=44100, size=-16, channels=1)
                _ready = pygame.mixer.get_init() is not None
            except pygame.error:
                _ready = False
        if not _ready:
            return
        fs, _, channels = pygame.mixer.get_init()
        n = int(fs * ms / 1000)
        t = np.arange(n) / fs
        fade = np.minimum(1.0, np.minimum(np.arange(n), n - np.arange(n)) / (0.01 * fs))
        wave = (np.sin(2 * np.pi * freq * t) * fade * volume * 32767).astype(np.int16)
        if channels > 1:
            wave = np.repeat(wave[:, None], channels, axis=1)
        pygame.sndarray.make_sound(np.ascontiguousarray(wave)).play()
    except Exception:
        pass
