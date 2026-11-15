"""pygame window: arena on the left, dashboard on the right."""
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
from pipeline.controller import Controller  # noqa: E402
from sinks.base import Sink  # noqa: E402
from sinks.sim_sink import SimSink  # noqa: E402
from sources.base import Source  # noqa: E402

from pipeline import ssvep_data  # noqa: E402

from . import sound, theme  # noqa: E402
from .arena_view import ArenaView  # noqa: E402
from .chase_view import ChaseView  # noqa: E402
from .dashboard import Dashboard  # noqa: E402
from .decoding_view import DecodingView  # noqa: E402
from .theme import Fonts  # noqa: E402
from .stimulus import (CalibrationRun, Stimulus, StimulusLayout, box_size, draw_boxes,  # noqa: E402
                       draw_calibration, draw_fitting, draw_warning)
from .training_panel import TrainingPanel  # noqa: E402

_DIRECTION_KEYS = {
    "up": (pygame.K_UP, pygame.K_w),
    "down": (pygame.K_DOWN, pygame.K_s),
    "left": (pygame.K_LEFT, pygame.K_a),
    "right": (pygame.K_RIGHT, pygame.K_d),
}
_ARROWS_ONLY = {"up": pygame.K_UP, "down": pygame.K_DOWN, "left": pygame.K_LEFT, "right": pygame.K_RIGHT}
# Training labels: top-row digits or keypad digits. Held state is reported as "0".."4".
_LABEL_KEYS = {str(n): (getattr(pygame, f"K_{n}"), getattr(pygame, f"K_KP{n}")) for n in range(10)}
_DIRECTIONS = ("up", "down", "left", "right")
# Extra held keys some sources understand (mock EEG: E = eyes closed, J = jaw clench).
_EXTRA_KEYS = {"eyes": pygame.K_e, "jaw": pygame.K_j}

HeldProvider = Callable[[float], frozenset]


class App:
    def __init__(self, source: Source, controller: Controller, sinks: Sequence[Sink],
                 sim: Optional[SimSink] = None, size: tuple[int, int] = config.WINDOW_SIZE,
                 view: str = config.ARENA_VIEW) -> None:
        self.source = source
        self.controller = controller
        self.sinks = list(sinks)
        self.sim = sim
        self.size = size
        self.fonts = Fonts()
        self.arena_view = ArenaView(self.fonts)
        self.chase_view = ChaseView(self.fonts)
        if view not in ("top", "chase"):
            raise ValueError(f"unknown view {view!r}")
        self.view = view
        self.dashboard = Dashboard(self.fonts)
        self.training_panel = TrainingPanel(self.fonts)
        self.training = False
        self._clear_armed_until = 0.0
        self.fps = 0.0
        self.now = time.monotonic()
        self.running = False
        self.screen: Optional[pygame.Surface] = None
        self._toast: Optional[tuple[str, float, tuple]] = None
        # SSVEP flicker stimulus (EEG sources in ssvep mode only)
        self.stimulus = Stimulus() if getattr(source, "supports_stimulus", False) else None
        self.warning_open = False
        self._after_warning: Optional[str] = None
        self.calibration: Optional[CalibrationRun] = None
        self.fitting: Optional[dict] = None   # calibration being fitted in a worker thread
        self.decoding_view = DecodingView(self.fonts)
        self.show_decoding = False            # I: how the commands are told apart, live
        self._cal_segment = -1
        self._cal_flips: list[tuple[float, int]] = []
        self.vsync = False
        self.refresh_hz: Optional[float] = None
        self.dropped_frames = 0
        self._flip_times: list[float] = []
        self._headless = os.environ.get("SDL_VIDEODRIVER") == "dummy"
        msg = getattr(source, "load_message", None)
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
        pygame.display.set_caption(f"neurorover — {self.source.name}")
        pygame.key.set_repeat()  # no key repeat: we read held state directly

    def run(self, duration: Optional[float] = None, max_frames: Optional[int] = None,
            screenshot: Optional[str] = None, held_provider: Optional[HeldProvider] = None) -> None:
        """Main loop. ``duration``/``max_frames``/``held_provider`` exist for headless smoke tests."""
        self._open_window()
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
                self._update_recording(held)
                self._update_calibration(now)
                self.controller.tick(now)
                if self.sim is not None:
                    self.sim.update(dt)
                self.draw()
                pygame.display.flip()
                if self.stimulus is not None:
                    if self.calibration is not None and self.stimulus.enabled:
                        self._cal_flips.append((time.monotonic(), self.stimulus.frame))  # frame just shown
                    self.stimulus.advance()  # flicker advances once per displayed frame
                    self._track_refresh(time.perf_counter())
                # With vsync the flip already paces us at the display's refresh rate.
                clock.tick() if self.vsync else clock.tick(config.FPS)
                self.fps = clock.get_fps()
                frames += 1
                if duration is not None and now - t0 >= duration:
                    break
                if max_frames is not None and frames >= max_frames:
                    break
            if screenshot:
                self.save_screenshot(screenshot)
        finally:
            pygame.display.quit()

    # -- input ---------------------------------------------------------------------------
    def _held_from_keyboard(self) -> set[str]:
        keys = pygame.key.get_pressed()
        ctrl = pygame.key.get_mods() & pygame.KMOD_CTRL
        held = set()
        table = {d: (k,) for d, k in _ARROWS_ONLY.items()} if ctrl else _DIRECTION_KEYS
        for direction, codes in table.items():
            if any(keys[c] for c in codes):
                held.add(direction)
        for label, codes in _LABEL_KEYS.items():
            if any(keys[c] for c in codes):
                held.add(label)
        for name, code in _EXTRA_KEYS.items():
            if keys[code]:
                held.add(name)
        return held

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
        """A user-facing warning if the display can't run the flicker accurately, else None."""
        if self.stimulus is None or self._headless:
            return None
        if not self.vsync:
            return "VSYNC UNAVAILABLE: flicker timing will be imprecise and SSVEP may not work."
        if self.refresh_hz is None:
            return None
        if abs(self.refresh_hz - config.SSVEP_REFRESH_HZ) > config.SSVEP_REFRESH_TOLERANCE_HZ:
            return (f"DISPLAY IS {self.refresh_hz:.0f} Hz: the flicker needs {config.SSVEP_REFRESH_HZ:.0f} Hz. "
                    "Windows: Settings > System > Display > Advanced display > Choose a refresh rate > 60 Hz.")
        return None

    def _update_recording(self, held) -> None:
        if not self.source.supports_training:
            return
        label = None
        if self.training:
            keys = getattr(self.source, "label_keys", config.GESTURE_KEYS)
            digits = sorted(int(h) for h in held if h in _LABEL_KEYS and int(h) in keys)
            if digits:
                label = keys[digits[0]]
        self.source.set_recording(label)

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
            if event.key == pygame.K_ESCAPE:
                if self.calibration is not None:
                    self.cancel_calibration()
                elif self.fitting is not None:
                    self.fitting = None   # the worker finishes on its own; its result is dropped
                    self.source.cancel_calibration()
                    self.controller.set_paused(False)
                    self.toast("Calibration fit cancelled (kept the previous decoder)")
                else:
                    self.running = False
            elif event.key == pygame.K_i:
                self.show_decoding = not self.show_decoding
            elif event.key == pygame.K_p:
                self.view = "top" if self.view == "chase" else "chase"
                self.toast("Chase view: rover faces up" if self.view == "chase" else "Top-down view")
            elif event.key == pygame.K_v:
                self.toggle_stimulus()
            elif event.key == pygame.K_c:
                self.start_calibration()
            elif event.key == pygame.K_r and hasattr(self.source, "measure_resistance"):
                if self.source.measure_resistance():
                    n = len(getattr(self.source, "links", ())) or 1
                    self.toast(f"Measuring electrode contact (~{5 * n} s, one headset at a time; rover stopped)")
                else:
                    self.toast(getattr(self.source, "resist_unavailable", None)
                               or "Contact can be re-measured once the headband is streaming", error=True)
            elif event.key == pygame.K_F5 and self.sim is not None:
                self.sim.reset()
                self.toast("Rover reset")
            elif event.key == pygame.K_F12:
                path = config.SCREENSHOTS_DIR / f"neurorover_{datetime.now():%Y%m%d_%H%M%S}.png"
                self.save_screenshot(path)
                self.toast(f"Saved {path.name}")
            elif event.key == pygame.K_t:
                self.toggle_training()
            elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self.fit_model()
            elif event.key == pygame.K_s and event.mod & pygame.KMOD_CTRL:
                self.save_model()
            elif event.key == pygame.K_x:
                self.clear_training_data()

    # -- SSVEP stimulus + calibration ------------------------------------------------------
    def toggle_stimulus(self) -> None:
        if self.stimulus is None:
            self.toast("The flicker stimulus needs an EEG source in SSVEP mode", error=True)
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
        if not hasattr(self.source, "calibration_segments"):
            self.toast("Calibration (C) is for EEG sources", error=True)
            return
        if self.calibration is not None:
            self.cancel_calibration()
            return
        if self.fitting is not None:
            self.toast("Still fitting the last calibration…", error=True)
            return
        if self.training:
            self.toast("Leave training mode (T) first: C runs its own guided gesture recording", error=True)
            return
        bad = self.source.bad_electrodes(decode_only=True) if hasattr(self.source, "bad_electrodes") else []
        if bad:
            notes = ", ".join(f"{p} {self.source.quality_note.get(p, '')}".strip() for p in bad)
            self.toast(f"Can't calibrate with a bad electrode: {notes}. Reseat it first.", 6.0, error=True)
            return
        if self.stimulus is not None:
            if not self.stimulus.warning_accepted:
                self.warning_open, self._after_warning = True, "calibrate"
                return
            self.stimulus.enabled = True
        self.calibration = CalibrationRun(self.source.calibration_segments(), time.monotonic())
        self._cal_segment = -1
        self._cal_flips = []
        self.source.start_recording()
        self.controller.set_paused(True)
        missing = self._missing_devices()
        note = f" Without {', '.join(missing)} (not streaming)." if missing else ""
        self.toast(f"Calibration: follow the prompts and beeps (~{self.calibration.total_s:.0f} s). "
                   f"Esc cancels.{note}", 6.0 if missing else 4.0)

    def _missing_devices(self) -> list[str]:
        """Configured EEG devices with no data right now (calibrating without them is allowed)."""
        names = getattr(self.source, "device_names", ())
        if len(names) < 2:
            return []
        here = set(getattr(self.source, "present_devices", ()))
        return [self.source.device_labels.get(n, n) for n in names if n not in here]

    def cancel_calibration(self) -> None:
        self.source.stop_recording()
        self.source.cancel_calibration()
        self.calibration = None
        self.controller.set_paused(False)
        self.toast("Calibration cancelled")

    def _update_calibration(self, now: float) -> None:
        self._poll_fit()
        if self.calibration is None:
            return
        cur = self.calibration.current(now)
        if cur is not None:
            if cur[0] != self._cal_segment:  # a beep at every step, so eyes-closed steps work too
                self._cal_segment = cur[0]
                sound.beep(660 if cur[0] == 0 else 880)
            self.source.set_calibration_label(cur[1])
            return
        run, self.calibration = self.calibration, None
        sound.beep(1175, 300)
        rec = self.source.stop_recording()
        timeline = self.source.calibration_timeline() if hasattr(self.source, "calibration_timeline") else None
        saved = self._save_calibration_recording(run, rec, timeline)
        self._start_fit(rec, timeline, saved)

    def _start_fit(self, rec, timeline, saved: Optional[Path]) -> None:
        """Fit in a worker thread (a few seconds) so the window stays live; the rover stays paused."""
        job = {"done": False, "cal": None, "error": None, "saved": saved, "progress": ("starting", 0.0)}

        def work() -> None:
            try:
                job["cal"] = self.source.fit_calibration(
                    rec, timeline, progress=lambda what, frac: job.__setitem__("progress", (what, frac)))
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
            self.source.cancel_calibration()
            self.toast(f"Calibration failed: {job['error']}", 8.0, error=True)
            return
        cal = self.source.apply_calibration(job["cal"])
        self.toast(f"Calibrated: {cal.describe()}. Saved{' + recording' if job['saved'] else ''}.", 10.0)

    def _save_calibration_recording(self, run: CalibrationRun, rec, timeline) -> Optional[Path]:
        """Raw EEG + segments + flicker frame times, for offline method comparison."""
        if rec is None:
            return None
        if timeline:
            segments = list(timeline)
        else:
            segments, t = [], run.start
            for label, dur, _ in run.segments:
                segments.append((label, t, t + dur))
                t += dur
        try:
            return ssvep_data.save_recording(
                mode=self.source.mode, user=self.source.user, fs=self.source.fs, positions=self.source.positions,
                t=rec[0], x=rec[1], segments=segments, flips=self._cal_flips,
                frames=self.stimulus.frames if self.stimulus else {}, refresh_hz=self.refresh_hz,
                extra={"vsync": self.vsync, "window_s": self.source.window_s},
                present=rec[2] if len(rec) > 2 else None,
                devices=getattr(self.source, "device_positions", None),
                models_dir=self.source.models_dir)
        except Exception as exc:
            print(f"[neurorover] couldn't save the calibration recording: {exc}")
            return None

    # -- training actions -------------------------------------------------------------
    def toggle_training(self) -> None:
        if not self.source.supports_training:
            self.toast(f"Training needs an EMG source (this is '{self.source.name}')", error=True)
            return
        self.training = not self.training
        self.controller.set_paused(self.training)
        if not self.training:
            self.source.set_recording(None)
        self.toast("Training mode: rover paused" if self.training else "Training mode off: driving")

    def fit_model(self) -> None:
        if not self.source.supports_training:
            return
        try:
            report = self.source.trainer.fit()
            self.toast(f"Model {report.describe()}", 4.0)
        except ValueError as exc:
            self.toast(f"Can't fit yet: {exc}", 4.0, error=True)

    def save_model(self) -> None:
        if not self.source.supports_training:
            return
        try:
            path = self.source.trainer.save()
            self.toast(f"Saved models/{path.name} (+ training data)", 4.0)
        except ValueError as exc:
            self.toast(f"Can't save yet: {exc}", 4.0, error=True)

    def clear_training_data(self) -> None:
        if not (self.source.supports_training and self.training):
            return
        now = time.monotonic()
        if now < self._clear_armed_until:
            self.source.trainer.clear()
            self._clear_armed_until = 0.0
            self.toast("Training data cleared (saved files untouched until Ctrl+S)")
        else:
            self._clear_armed_until = now + 2.0

    def key_hints(self) -> str:
        if getattr(self.source, "eeg", False):
            mock = self.source.name == "mock-eeg"
            if self.stimulus is not None:
                first = "Arrows gaze · E eyes shut · J jaw  ·  " if mock else "R contact  ·  "
                return f"{first}V flicker · C calibrate · I decoder  ·  P view  ·  Esc quit"
            first = "E eyes shut (→ F) · J jaw  ·  " if mock else "Close your eyes to drive  ·  R contact  ·  "
            return f"{first}C calibrate  ·  P view  ·  F5 reset  ·  Esc quit"
        if self.source.supports_training:
            first = "Arrows: virtual gesture  ·  " if self.source.name == "mock" else ""
            return f"{first}C calibrate gestures · I decoder  ·  T train  ·  P view  ·  Esc quit"
        return "WASD / arrows drive  ·  P view  ·  F5 reset rover  ·  F12 screenshot  ·  Esc quit"

    def toast(self, text: str, seconds: float = 2.0, error: bool = False) -> None:
        self._toast = (text, time.monotonic() + seconds, theme.BAD if error else theme.PANEL_HI)

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
        renderer = self.chase_view if self.view == "chase" else self.arena_view
        sim = self.sim.sim if self.sim else None
        if self.stimulus is not None and self.stimulus.enabled:
            # Flicker on: boxes at the window edges, rover in the middle, dashboard in the corners.
            lay = StimulusLayout.fit(surf.get_rect(), box_size(self.fonts), self.fonts.px(16))
            freqs = self.stimulus.frequencies(config.SSVEP_REFRESH_HZ)
            cur = self.calibration.current(time.monotonic()) if self.calibration else None
            highlight = cur[1] if cur and cur[1] in lay.boxes else None
            draw_boxes(surf, self.fonts, lay, self.stimulus, freqs, self.source.command_map,
                       self.controller.last_sent, highlight)
            view_r = lay.centre
            renderer.draw(surf, view_r, sim, self.banners())
            self.dashboard.draw_panels(surf, lay.panels, self)
        else:
            view_r, dash_r = self.layout()
            renderer.draw(surf, view_r, sim, self.banners())
            self.dashboard.draw(surf, dash_r, self)
        if self.show_decoding:
            self.decoding_view.draw(surf, view_r, self)
        if self.calibration is not None:
            draw_calibration(surf, self.fonts, view_r, self.calibration, time.monotonic())
        elif self.fitting is not None:
            draw_fitting(surf, self.fonts, view_r, *self.fitting["progress"])
        if self.training and self.source.supports_training:
            self.training_panel.draw(surf, view_r, self.source, time.monotonic() < self._clear_armed_until)
        self._draw_toast(surf, view_r)

    def banners(self) -> list[tuple[str, tuple[int, int, int]]]:
        out = []
        problem = self.refresh_problem()
        if problem:
            out.append((problem, theme.BAD))
        st = self.source.status()
        if not st.ok:
            out.append((f"{self.source.name.upper()}: {st.message}", theme.WARN if st.pending else theme.BAD))
        return out

    def fps_text(self) -> str:
        text = f"FPS {self.fps:.0f}"
        if self.stimulus is not None:
            if self.vsync:
                text += f"  ·  vsync {self.refresh_hz:.0f} Hz" if self.refresh_hz else "  ·  vsync"
            elif not self._headless:
                text += "  ·  NO vsync"
            if self.dropped_frames:
                text += f"  ·  {self.dropped_frames} dropped"
        return text

    def _draw_toast(self, surf, area: pygame.Rect) -> None:
        if not self._toast:
            return
        text, until, fill = self._toast
        if time.monotonic() > until:
            self._toast = None
            return
        f = self.fonts
        t = f.render(text, 24, theme.TEXT, bold=True)
        r = t.get_rect()
        r.inflate_ip(f.px(28), f.px(14))
        r.midbottom = (area.centerx, area.bottom - f.px(20))
        theme.panel(surf, r, fill=fill, radius=f.px(8))
        surf.blit(t, t.get_rect(center=r.center))

    def save_screenshot(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(self.screen, str(path))
        return path
