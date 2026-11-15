"""A live camera for the middle of the rover window (main.py --camera): an iPhone through macOS
Continuity Camera, OBS Virtual Camera, or any webcam, pointed at the real rover. The driver then
watches the real rover while resting (looking at the centre) instead of the simulated one.

ON HOLD (2026-11-15): OpenCV's macOS wheels (4.14 and 5.0) bundle their own SDL2, which clashes with
pygame's in one process (objc "implemented in both"; "may cause mysterious crashes"). OpenCV is only
imported with --camera, so nothing else is affected. To pick it up again: capture in a separate
process, or via AVFoundation (pyobjc) instead of OpenCV. Not in requirements.txt yet.

Frames are read on a background thread (OpenCV), shrunk and converted to RGB there, so the flicker's
frame timing isn't disturbed: the window only blits the newest frame. A camera is opened on the
main thread, because macOS only asks for camera permission from there.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable, Optional

import numpy as np

MAX_SIZE = (1280, 720)      # frames are shrunk to fit this (enough for the middle of the window)
MAX_INDEX = 6               # K cycles through cameras 0..MAX_INDEX-1
STALE_S = 2.0               # no new frame for this long = the camera stopped
PERMISSION_HINT = ("if macOS asked, allow camera access for your terminal app "
                   "(System Settings > Privacy & Security > Camera)")


def _open_cv(index: int):
    """(capture, None) or (None, why not)."""
    try:
        import cv2
    except ImportError:
        return None, "OpenCV isn't installed (pip install opencv-python-headless)"
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION) if hasattr(cv2, "CAP_AVFOUNDATION") else cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        return None, f"camera {index} isn't available: {PERMISSION_HINT}; K tries the next camera"
    return cap, None


def _prepare(frame: np.ndarray) -> np.ndarray:
    """BGR camera frame -> RGB, no bigger than MAX_SIZE."""
    import cv2

    h, w = frame.shape[:2]
    k = min(1.0, MAX_SIZE[0] / w, MAX_SIZE[1] / h)
    if k < 1.0:
        frame = cv2.resize(frame, (max(1, int(w * k)), max(1, int(h * k))), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


class Camera:
    def __init__(self, index: int = 0, opener: Callable = _open_cv, prepare: Callable = _prepare,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.index = index
        self._opener = opener
        self._prepare = prepare
        self.clock = clock
        self._cap = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._frame_t: Optional[float] = None
        self.frames = 0
        self.fps = 0.0
        self.error: Optional[str] = None
        self._opened_t: Optional[float] = None

    def start(self) -> None:
        """Open camera ``index`` (call from the main thread) and start reading it."""
        self.stop()
        cap, why = self._opener(self.index)
        with self._lock:
            self._frame, self._frame_t, self.frames, self.fps = None, None, 0, 0.0
        self.error = why
        if cap is None:
            return
        self._cap = cap
        self._opened_t = self.clock()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(cap,), name="camera", daemon=True)
        self._thread.start()

    def _run(self, cap) -> None:
        times: deque = deque(maxlen=30)
        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue
            try:
                frame = self._prepare(frame)
            except Exception:
                continue
            now = self.clock()
            times.append(now)
            with self._lock:
                self._frame, self._frame_t = frame, now
                self.frames += 1
                if len(times) > 1 and times[-1] > times[0]:
                    self.fps = (len(times) - 1) / (times[-1] - times[0])

    def latest(self) -> tuple[Optional[np.ndarray], Optional[float]]:
        """The newest frame (RGB, [h, w, 3]) and when it arrived."""
        with self._lock:
            return self._frame, self._frame_t

    def status(self) -> tuple[bool, str]:
        """(live, what to show)."""
        if self.error:
            return False, self.error
        now = self.clock()
        _, t = self.latest()
        if t is None:
            waited = now - (self._opened_t or now)
            if waited > 5.0:
                return False, f"camera {self.index} sends no picture: {PERMISSION_HINT}; K tries the next camera"
            return False, f"starting camera {self.index}…"
        if now - t > STALE_S:
            return False, f"camera {self.index} stopped sending pictures (phone locked or moved away?)"
        return True, f"camera {self.index} · {self.fps:.0f} fps"

    def next(self) -> None:
        """Switch to the next camera (K): the Mac's own, the iPhone, OBS... in whatever order macOS gives."""
        self.index = (self.index + 1) % MAX_INDEX
        self.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
        self._cap = None
