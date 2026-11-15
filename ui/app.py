"""The rover window: arena (top-down or chase cam) and dashboard; with the SSVEP flicker on, four
boxes at the window edges with the arena in the middle (look at the rover = rest)."""
from __future__ import annotations

import os
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Sequence

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402

import config  # noqa: E402
from eeg import cues  # noqa: E402
from rover.camera import Camera  # noqa: E402
from rover.controller import Controller  # noqa: E402
from rover.sim import SimSink, Sink  # noqa: E402
from rover.source import Source  # noqa: E402

from . import sound, theme  # noqa: E402
from .arena_view import ArenaView  # noqa: E402
from .camera_view import CameraView  # noqa: E402
from .chase_view import ChaseView  # noqa: E402
from .dashboard import Dashboard  # noqa: E402
from .decoding_view import DecodingView  # noqa: E402
from .drive_view import DriveView  # noqa: E402
from .stimulus import (Stimulus, StimulusLayout, box_labels, box_size, draw_boxes,  # noqa: E402
                       draw_calibration, draw_fitting, draw_warning, panels_clear_of)
from .theme import Fonts  # noqa: E402

_DIRECTION_KEYS = {
    "up": (pygame.K_UP, pygame.K_w),
    "down": (pygame.K_DOWN, pygame.K_s),
    "left": (pygame.K_LEFT, pygame.K_a),
    "right": (pygame.K_RIGHT, pygame.K_d),
}

HeldProvider = Callable[[float], frozenset]


class App:
    def __init__(self, source: Source, controller: Controller, sinks: Sequence[Sink],
                 sim: Optional[SimSink] = None, size: tuple[int, int] = config.WINDOW_SIZE,
                 view: str = config.ARENA_VIEW, camera: Optional[Camera] = None) -> None:
        if view not in ("top", "chase", "camera"):
            raise ValueError(f"unknown view {view!r}")
        if view == "camera" and camera is None:
            raise ValueError("the camera view needs a camera")
        self.source = source
        self.controller = controller
        self.sinks = list(sinks)
        self.sim = sim
        self.size = size
        self.view = view
        self.fonts = Fonts()
        self.arena_view = ArenaView(self.fonts)
        self.chase_view = ChaseView(self.fonts)
        self.camera = camera                  # live camera in the middle instead of the simulated arena
        self.camera_view = CameraView(self.fonts)
        self.dashboard = Dashboard(self.fonts)
        self.decoding_view = DecodingView(self.fonts)
        self.drive_view = DriveView(self.fonts, self.dashboard)     # the middle when there's no simulated arena
        self.show_decoding = False            # I: how the boxes are told apart, live
        self.fps = 0.0
        self.now = time.monotonic()
        self.running = False
        self.screen: Optional[pygame.Surface] = None
        self._toast: Optional[tuple[str, float, tuple]] = None
        self.stimulus = Stimulus() if source.eeg else None
        self.warning_open = False
        self._after_warning: Optional[str] = None
        self.calibration: Optional[cues.Run] = None   # the cued calibration being shown
        self.fitting: Optional[dict] = None   # calibration being fitted in a worker thread
        self._cal_flips: list[tuple[float, int]] = []
        self.vsync = False
        self.refresh_hz: Optional[float] = None
        self.dropped_frames = 0
        self._flip_times: list[float] = []
        self._headless = os.environ.get("SDL_VIDEODRIVER") == "dummy"
        msg = getattr(source, "load_message", "")
        if msg:
            self.toast(msg[0].upper() + msg[1:], 4.0)

    # -- lifecycle -----------------------------------------------------------------------
    def _open_window(self) -> None:
        pygame.display.init()
        pygame.font.init()
        w, h = self.size
        if not self._headless:
            try:  # never open a window bigger than the desktop
                dw, dh = pygame.display.get_desktop_sizes()[0]
                k = min(1.0, 0.95 * dw / w, 0.9 * dh / h)
                w, h = int(w * k), int(h * k)
            except Exception:
                pass
        self.screen = None
        if self.stimulus is not None and not self._headless:
            # Frame-counted flicker needs vsync; in pygame 2 that requires the SCALED renderer.
            try:
                self.screen = pygame.display.set_mode((w, h), pygame.RESIZABLE | pygame.SCALED, vsync=1)
                self.vsync = True
            except pygame.error:
                self.screen, self.vsync = None, False
        if self.screen is None:
            self.screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
        pygame.display.set_caption(f"neurover — {self.source.name}")
        pygame.key.set_repeat()  # no key repeat: held state is read directly

    def run(self, duration: Optional[float] = None, max_frames: Optional[int] = None,
            screenshot: Optional[str] = None, held_provider: Optional[HeldProvider] = None) -> None:
        """Main loop. ``duration``/``max_frames``/``held_provider`` exist for headless tests."""
        self._open_window()
        if self.camera is not None:
            self.camera.start()               # on the main thread: macOS asks for camera permission here
        clock = pygame.time.Clock()
        t0 = last = time.monotonic()
        frames = 0
        self.running = True
        try:
            while self.running:
                for event in pygame.event.get():
                    self._on_event(event)
                now = time.monotonic()
                dt, last = now - last, now
                self.now = now
                held = held_provider(now - t0) if held_provider else self._held_from_keyboard()
                self.source.on_input(frozenset(held))
                self._update_calibration(now)
                self.controller.tick(now)
                if self.sim is not None:
                    self.sim.update(dt)
                self.draw()
                pygame.display.flip()
                if self.stimulus is not None:
                    if self.calibration is not None and self.stimulus.enabled:
                        self._cal_flips.append((time.monotonic(), self.stimulus.frame))  # frame just shown
                    self.stimulus.advance()  # the flicker advances once per displayed frame
                    self._track_refresh(time.perf_counter())
                clock.tick() if self.vsync else clock.tick(config.FPS)   # with vsync the flip paces us
                self.fps = clock.get_fps()
                frames += 1
                if duration is not None and now - t0 >= duration:
                    break
                if max_frames is not None and frames >= max_frames:
                    break
            if screenshot:
                self.save_screenshot(screenshot)
        finally:
            if self.camera is not None:
                self.camera.stop()
            pygame.display.quit()

    # -- input ---------------------------------------------------------------------------
    def _held_from_keyboard(self) -> set[str]:
        keys = pygame.key.get_pressed()
        held = {d for d, codes in _DIRECTION_KEYS.items() if any(keys[c] for c in codes)}
        if keys[pygame.K_j]:
            held.add("jaw")                       # mock: clench the jaw (an artifact)
        return held

    def _on_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.KEYDOWN and self.warning_open:
            # The photosensitivity warning swallows every key except its own two.
            if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self.accept_warning()
            elif event.key == pygame.K_ESCAPE:
                self.warning_open, self._after_warning = False, None
                self.toast("Flicker cancelled")
        elif event.type == pygame.KEYDOWN:
            k = event.key
            if k == pygame.K_ESCAPE:
                if self.calibration is not None:
                    self.cancel_calibration()
                elif self.fitting is not None:
                    self.fitting = None           # the worker finishes on its own; its result is dropped
                    self.controller.set_paused(False)
                    self.toast("Calibration fit cancelled (kept the previous settings)")
                else:
                    self.running = False
            elif k == pygame.K_i and self.source.eeg:
                self.show_decoding = not self.show_decoding
            elif k == pygame.K_p:
                views = (["camera"] if self.camera is not None else []) + ["top", "chase"]
                self.view = views[(views.index(self.view) + 1) % len(views)]
                self.toast({"camera": "Camera", "top": "Top-down view", "chase": "Chase view: rover faces up"}[self.view])
            elif k == pygame.K_k and self.camera is not None:
                self.camera.next()
                self.toast(f"Camera {self.camera.index}")
            elif k == pygame.K_v:
                self.toggle_stimulus()
            elif k == pygame.K_c:
                self.start_calibration()
            elif k == pygame.K_SPACE and self.calibration is not None:
                self.calibration.skip(time.monotonic())        # get-ready / break: go on
            elif k == pygame.K_r and self.source.eeg:
                if self.source.measure_resistance():
                    n = len(self.source.feeds.devices)
                    self.toast(f"Measuring electrode contact (~{5 * n} s, one headset at a time; rover stops)")
                else:
                    self.toast("Contact can be measured once a headset is streaming", error=True)
            elif k == pygame.K_F5 and self.sim is not None:
                self.sim.reset()
                self.toast("Rover reset")
            elif k == pygame.K_F12:
                path = config.SCREENSHOTS_DIR / f"neurover_{datetime.now():%Y%m%d_%H%M%S}.png"
                self.save_screenshot(path)
                self.toast(f"Saved {path.name}")

    # -- flicker + calibration ---------------------------------------------------------------
    def toggle_stimulus(self) -> None:
        if self.stimulus is None:
            self.toast("The flicker is for EEG sources", error=True)
            return
        if self.stimulus.enabled:
            if self.calibration is not None:
                self.cancel_calibration()
            self.stimulus.enabled = False
            self.toast("Flicker off")
        elif not self.stimulus.warning_accepted:
            self.warning_open, self._after_warning = True, "stimulus"
        else:
            self.stimulus.enabled = True

    def accept_warning(self) -> None:
        self.warning_open = False
        self.stimulus.warning_accepted = True
        action, self._after_warning = self._after_warning, None
        self.stimulus.enabled = True
        if action == "calibrate":
            self.start_calibration()

    def start_calibration(self) -> None:
        if not self.source.eeg:
            self.toast("Calibration (C) is for EEG sources", error=True)
            return
        if self.calibration is not None:
            self.cancel_calibration()
            return
        if self.fitting is not None:
            self.toast("Still fitting the last calibration…", error=True)
            return
        if not self.stimulus.warning_accepted:
            self.warning_open, self._after_warning = True, "calibrate"
            return
        self.stimulus.enabled = True
        self.calibration = cues.Run(self.source.calibration_protocol(), time.monotonic(), name="ssvep calibration")
        self._cal_flips = []
        self.source.start_calibration()
        self.controller.set_paused(True)
        self._toast = None                      # the calibration card says it all; keep the cross clear

    def cancel_calibration(self) -> None:
        self.source.cancel_calibration()
        self.calibration = None
        self.controller.set_paused(False)
        self.toast("Calibration cancelled")

    def _update_calibration(self, now: float) -> None:
        self._poll_fit()
        run = self.calibration
        if run is None:
            return
        for step in run.update(now):     # low: look at the rover; middle: back to the cross; high: the box
            if step.kind == "away":
                sound.beep(440, 250)
            elif step.kind == "fixation" and run.protocol.away_s > 0:
                sound.beep(660)
            elif step.kind == "cue":
                sound.beep(880)
            elif step.kind in ("go", "break"):
                sound.beep(523, 350)
        if not run.finished:
            return
        self.calibration = None
        sound.beep(1175, 300)
        rec, path = self.source.finish_calibration(
            {**run.meta(), "flips": self._cal_flips, "refresh_hz": self.refresh_hz, "vsync": self.vsync})
        if rec is None:
            self.controller.set_paused(False)
            self.toast("Calibration failed: no EEG arrived", 6.0, error=True)
            return
        rec["path"] = path
        self._start_fit(rec)

    def _start_fit(self, rec: dict) -> None:
        """Fit in a worker thread (several seconds) so the window stays live; the rover stays paused."""
        job = {"done": False, "settings": None, "error": None, "progress": ("starting", 0.0)}

        def work() -> None:
            try:
                job["settings"] = self.source.fit_calibration(
                    rec, progress=lambda what, frac: job.__setitem__("progress", (what, frac)))
            except Exception as exc:
                if not isinstance(exc, ValueError):
                    traceback.print_exc()
                job["error"] = exc
            finally:
                job["done"] = True

        self.fitting = job
        threading.Thread(target=work, name="calibration-fit", daemon=True).start()

    def _poll_fit(self) -> None:
        job = self.fitting
        if job is None or not job["done"]:
            return
        self.fitting = None
        self.controller.set_paused(False)
        if job["error"] is not None:
            self.toast(f"Calibration failed: {job['error']}", 20.0, error=True)
            return
        s = self.source.apply_calibration(job["settings"])
        self.toast(f"Calibrated: {s.describe()}", 10.0)

    def _track_refresh(self, t: float) -> None:
        """Measure the real refresh rate from vsync'd flips; count dropped frames while flickering."""
        self._flip_times.append(t)
        if len(self._flip_times) > 240:
            del self._flip_times[:-240]
        if len(self._flip_times) >= 90:
            gaps = sorted(b - a for a, b in zip(self._flip_times[20:], self._flip_times[21:]))
            if gaps:
                self.refresh_hz = 1.0 / max(1e-6, gaps[len(gaps) // 2])
        if self.refresh_hz and self.stimulus is not None and self.stimulus.enabled and len(self._flip_times) > 1:
            if self._flip_times[-1] - self._flip_times[-2] > 1.5 / self.refresh_hz:
                self.dropped_frames += 1

    def refresh_problem(self) -> Optional[str]:
        """A warning if the display can't run the flicker accurately, else None."""
        if self.stimulus is None or self._headless:
            return None
        if not self.vsync:
            return "Vsync unavailable: flicker timing will be imprecise and SSVEP may not work."
        if self.refresh_hz is None:
            return None
        if abs(self.refresh_hz - config.SSVEP_REFRESH_HZ) > config.SSVEP_REFRESH_TOLERANCE_HZ:
            return (f"Display is {self.refresh_hz:.0f} Hz: the flicker needs {config.SSVEP_REFRESH_HZ:.0f} Hz "
                    "(set the display to 60 Hz).")
        return None

    def key_hints(self) -> list[list[tuple[str, str]]]:
        """Groups of (keys, what they do); "A / B" = either key."""
        if self.source.eeg:
            first = [("Arrows", "gaze"), ("J", "jaw")] if self.source.name == "mock" else [("R", "contact")]
            cam = [("K", "camera")] if self.camera is not None else []
            return [first, [("V", "flicker"), ("C", "calibrate"), ("I", "decoder")],
                    [("P", "view"), *cam, ("F5", "reset")], [("Esc", "quit")]]
        return [[("WASD / Arrows", "drive")], [("P", "view")], [("F5", "reset rover")], [("F12", "screenshot")],
                [("Esc", "quit")]]

    def toast(self, text: str, seconds: float = 2.0, error: bool = False) -> None:
        """A message over the arena. Long ones wrap and stay up long enough to read (~15 characters a
        second); errors also go to the console, so they're in the log."""
        seconds = max(seconds, len(text) / 15.0)
        self._toast = (text, time.monotonic() + seconds, theme.BAD if error else theme.PANEL_HI)
        if error:
            print(f"[neurover] {text}", flush=True)

    # -- drawing -------------------------------------------------------------------------
    def layout(self) -> tuple[pygame.Rect, pygame.Rect]:
        w, h = self.screen.get_size()
        dash_w = max(config.DASHBOARD_MIN_W, round(w * config.DASHBOARD_FRACTION))
        dash_w = min(dash_w, w - 200)
        return pygame.Rect(0, 0, w - dash_w, h), pygame.Rect(w - dash_w, 0, dash_w, h)

    def draw(self) -> None:
        surf = self.screen
        w, h = surf.get_size()
        self.fonts.set_scale(min(w / config.WINDOW_SIZE[0], h / config.WINDOW_SIZE[1]))
        if self.warning_open:
            draw_warning(surf, self.fonts)
            return
        surf.fill(theme.BG)

        boxes = None
        if self.stimulus is not None and self.stimulus.enabled:
            # Flicker on: boxes at the window edges, rover in the middle, dashboard in the corners.
            lay = StimulusLayout.fit(surf.get_rect(), box_size(self.fonts), self.fonts.px(16))
            boxes = lay.boxes
            freqs = self.stimulus.frequencies(config.SSVEP_REFRESH_HZ)
            st = self.calibration.step if self.calibration is not None and not self.calibration.finished else None
            highlight = st.label if st is not None and st.kind == "cue" and st.label in lay.boxes else None
            blank = st is not None and st.kind in ("pause", "break")      # rest the eyes (not used for fitting)
            draw_boxes(surf, self.fonts, lay, self.stimulus, freqs, self.source.command_map,
                       self.controller.last_sent, highlight, blank)
            view_r = lay.centre
            self._draw_middle(surf, view_r)
            # Small windows: the corner panels would run into the side boxes (and their frames and labels);
            # shorten them so the boxes are always drawn whole.
            labels = box_labels(self.fonts, lay, self.stimulus, freqs, self.source.command_map)
            frames = [b.inflate(self.fonts.px(22), self.fonts.px(22)) for k, b in lay.boxes.items()
                      if k in self.stimulus.patterns]
            self.dashboard.draw_panels(surf, panels_clear_of(lay.panels, [*labels.values(), *frames],
                                                             self.fonts.px(8)), self, drive=self._drive_shown())
        else:
            view_r, dash_r = self.layout()
            self._draw_middle(surf, view_r)
            self.dashboard.draw(surf, dash_r, self, drive=self._drive_shown())
        if self.show_decoding:
            self.decoding_view.draw(surf, view_r, self)
        if self.calibration is not None:
            draw_calibration(surf, self.fonts, view_r, self.calibration, time.monotonic(), boxes)
        elif self.fitting is not None:
            draw_fitting(surf, self.fonts, view_r, *self.fitting["progress"])
        self._draw_toast(surf, view_r)

    def _draw_middle(self, surf, rect: pygame.Rect) -> None:
        """What the driver looks at to rest: the live camera, or the simulated arena (top-down / chase)."""
        if self.view == "camera":
            self.camera_view.draw(surf, rect, self.camera, self.banners())
            return
        if self.sim is None:
            self._draw_no_sim(surf, rect)
            return
        renderer = self.chase_view if self.view == "chase" else self.arena_view
        renderer.draw(surf, rect, self.sim.sim, self.banners())

    def _drive_shown(self) -> bool:
        """No simulated arena and nothing else in the middle: it shows the drive view (command, box
        scores, rover link). Not while calibrating or fitting, nor under the decoder view (I): then the
        middle is blank as before and the dashboard is whole."""
        return (self.sim is None and self.view != "camera" and self.calibration is None and self.fitting is None
                and not self.show_decoding)

    def _draw_no_sim(self, surf, rect: pygame.Rect) -> None:
        """No simulated arena (driving the real rover): the drive view, or blank; warnings on top."""
        if self._drive_shown():
            self.drive_view.draw(surf, rect, self)
        self.arena_view._banners(surf, rect, self.banners())

    def banners(self) -> list[tuple[str, tuple[int, int, int]]]:
        out = []
        problem = self.refresh_problem()
        if problem:
            out.append((problem, theme.BAD))
        st = self.source.status()
        if not st.ok:
            out.append((f"{self.source.name.upper()}: {st.message}", theme.WARN if st.pending else theme.BAD))
        return out

    def frame_status(self) -> list[tuple[str, str]]:
        """Frame timing for the status bar: (text, "ok" | "warn" | "bad") parts."""
        parts = [(f"{self.fps:.0f} fps", "ok")]
        if self.stimulus is not None:
            if self.vsync:
                parts.append((f"vsync {self.refresh_hz:.0f} Hz" if self.refresh_hz else "vsync", "ok"))
            elif not self._headless:
                parts.append(("no vsync", "bad"))
            if self.dropped_frames:
                parts.append((f"{self.dropped_frames} dropped", "warn"))
        return parts

    def fps_text(self) -> str:
        return "   ".join(text for text, _ in self.frame_status())

    def _draw_toast(self, surf, area: pygame.Rect) -> None:
        if not self._toast:
            return
        text, until, fill = self._toast
        if time.monotonic() > until:
            self._toast = None
            return
        f = self.fonts
        pad = f.px(16)
        max_w = min(area.w - f.px(40), f.px(780)) - 2 * pad       # a readable measure, however wide the view
        size = 21 if f.width(text, 21, face=theme.DEMI) <= max_w else 18     # long messages: smaller, wrapped
        lines = theme.wrap(f, text, size, max_w, max_lines=12, face=theme.DEMI)
        ink = theme.INK if fill == theme.BAD else theme.TEXT
        surfs = [f.render(line, size, ink, face=theme.DEMI) for line in lines]
        line_h = max(s.get_height() for s in surfs)
        r = pygame.Rect(0, 0, max(s.get_width() for s in surfs) + 2 * pad, line_h * len(surfs) + f.px(16))
        r.midbottom = (area.centerx, area.bottom - f.px(20))
        r.top = max(r.top, area.top + f.px(10))
        theme.panel(surf, r, fill=fill, border=None if fill == theme.BAD else theme.EDGE, radius=f.px(8))
        for i, s in enumerate(surfs):
            surf.blit(s, s.get_rect(midtop=(r.centerx, r.top + f.px(8) + i * line_h)))

    def save_screenshot(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(self.screen, str(path))
        return path
