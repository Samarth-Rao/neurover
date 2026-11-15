"""pygame window with three views (Tab switches):

LIVE      every electrode's feed, device by device: trace, spectrum, numbers, status. Space records.
COMPARE   saved recordings side by side, electrode by electrode: overlaid spectra and one metric
          (←/→) as bars, one colour per recording. P (or ▶) plays one back.
PLAYBACK  a recording through the LIVE view, with play/pause, seek, speed and a timeline.
TRIALS    a cued-trial recording's per-trial analysis (T records one from LIVE; see trials_ui.py).

Click an electrode's name (LIVE or PLAYBACK) to say where it is on the head: live, that's where it
is from now on (remembered in scope_placement.json); in playback, it corrects that recording.
"""
from __future__ import annotations

import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import numpy as np  # noqa: E402
import pygame  # noqa: E402

import config  # noqa: E402
from ui import theme  # noqa: E402
from ui.theme import Fonts  # noqa: E402

from eeg import cues, recording  # noqa: E402
from eeg.devices import Feeds  # noqa: E402

from . import analysis, placement  # noqa: E402
from .playback import SPEEDS, Playback  # noqa: E402
from .trials_ui import CLASS_TEXT, TrialsUI, class_color  # noqa: E402

DESIGN = (1600, 900)
SPANS_S = (2.0, 5.0, 10.0, 20.0)
SCALES_UV = (10, 20, 50, 100, 200, 500, 1000, 5000, 50000)    # trace half-height
REC_COLORS = [(64, 170, 255), (255, 152, 40), (52, 211, 120), (204, 120, 255), (255, 110, 180), (80, 220, 220)]
STATS_S = 4.0               # the numbers + spectrum beside each trace cover the last 4 s
STATS_EVERY_S = 0.25
SEEK_S = 2.0
SEVERITY = {"good": theme.GOOD, "warn": theme.WARN, "bad": theme.BAD, "none": theme.DIM}
PSD_SPAN_DB = 45.0
_MOCK_GAZE_KEYS = ((pygame.K_1, "up"), (pygame.K_2, "down"), (pygame.K_3, "left"), (pygame.K_4, "right"))

Script = Callable[["ScopeApp", float], None]


def _state_color(state: str):
    return theme.GOOD if state == "streaming" else theme.BAD if state == "error" else theme.WARN


def _clock(seconds: float) -> str:
    s = int(max(0, seconds))
    return f"{s // 60}:{s % 60:02d}"


def _clock_tenths(seconds: float) -> str:
    s = max(0.0, seconds)
    return f"{int(s // 60)}:{s % 60:04.1f}"


def _ohm_text(ohm: float) -> str:
    if not math.isfinite(ohm):
        return "open"
    return f"{ohm / 1e6:.1f}MΩ" if ohm >= 1e5 else f"{ohm / 1e3:.0f}kΩ"


class ScopeApp(TrialsUI):
    def __init__(self, feeds: Feeds, rec_dir: Path = recording.RECORDINGS_DIR,
                 size: tuple[int, int] = DESIGN, view: str = "live",
                 placement_path: Optional[Path] = None, protocol: cues.Protocol = cues.Protocol()) -> None:
        if view not in ("live", "compare"):
            raise ValueError(f"unknown view {view!r}")
        self.feeds = feeds
        self.rec_dir = Path(rec_dir)
        self.placement_path = Path(placement_path) if placement_path else None   # None: don't remember moves
        self.size = size
        self.view = view
        self.fonts = Fonts()
        self.filtered = True
        self.span_i = 1
        self.scale_i = 2
        self.row_stats: dict[str, dict[tuple[str, int], dict]] = {"live": {}, "playback": {}}
        self._stats_at = -math.inf
        self.prompt: Optional[dict] = None       # name a recording
        self.picker: Optional[dict] = None       # where is this electrode?
        self.toast: Optional[tuple[str, float, tuple]] = None
        self.library: list[dict] = []
        self.cursor = 0
        self.selected: list[Path] = []
        self._slot: dict[Path, int] = {}
        self._analysis: dict[Path, tuple[float, dict]] = {}
        self.metric_i = 0
        self.playback: Optional[Playback] = None
        self.now = time.monotonic()
        self.running = False
        self.screen: Optional[pygame.Surface] = None
        self._hits: list[tuple[pygame.Rect, Callable[[], None]]] = []
        self._modal_hits: list[tuple[pygame.Rect, Callable[[], None]]] = []
        self._modal_box: Optional[pygame.Rect] = None
        self._timeline: Optional[pygame.Rect] = None
        self._dragging = False
        self._hand = False
        self._list_top = 0
        self._headless = os.environ.get("SDL_VIDEODRIVER") == "dummy"
        self._init_trials(protocol)
        if feeds.placement_note:
            self.flash(feeds.placement_note, theme.WARN, 6.0)

    # -- loop ---------------------------------------------------------------------------
    def _open_window(self) -> None:
        pygame.display.init()
        pygame.font.init()
        w, h = self.size
        if not self._headless:
            try:  # never bigger than the desktop
                dw, dh = pygame.display.get_desktop_sizes()[0]
                k = min(1.0, 0.95 * dw / w, 0.9 * dh / h)
                w, h = int(w * k), int(h * k)
            except Exception:
                pass
        self.screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
        pygame.display.set_caption("neurover scope")
        pygame.key.set_repeat()

    def run(self, duration: Optional[float] = None, screenshot: Optional[str] = None,
            script: Optional[Script] = None) -> None:
        self._open_window()
        self.feeds.start()
        self.refresh_library()
        clock = pygame.time.Clock()
        t0 = time.monotonic()
        self.running = True
        try:
            while self.running:
                for event in pygame.event.get():
                    self.on_event(event)
                self.now = time.monotonic()
                if script is not None:
                    script(self, self.now - t0)
                self._mock_keys()
                if self.cued is not None:
                    self._update_cued()
                self.feeds.poll()
                if self.playback is not None:
                    self.playback.update(self.now)
                self._update_stats()
                self.draw()
                pygame.display.flip()
                clock.tick(config.FPS)
                if duration is not None and self.now - t0 >= duration:
                    break
            if screenshot:
                self.save_screenshot(screenshot)
        finally:
            if self.feeds.recording:        # never lose a recording by quitting (a cued one keeps its trials)
                self.stop_recording(ask_name=False)
            self._close_trials_pool()
            self.feeds.stop()
            pygame.display.quit()

    # -- input --------------------------------------------------------------------------
    def on_event(self, event) -> None:
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.TEXTINPUT:
            box = self.prompt if self.prompt is not None else self.picker
            if box is not None and box.get("swallow") and time.monotonic() < box["swallow"][1]:
                key, box["swallow"] = box["swallow"][0], None
                if event.text.lower() == key:   # the key that opened this box, not the first letter typed
                    return
            if box is not None and len(box["text"]) < 48:
                box["text"] += "".join(ch for ch in event.text if ch.isprintable())
        elif event.type == pygame.KEYDOWN:
            self._on_key(event)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._on_click(event.pos)
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self._dragging = False
        elif event.type == pygame.MOUSEMOTION and self._dragging:
            self._seek_to_x(event.pos[0])
        elif event.type == pygame.MOUSEWHEEL and self.view == "compare" and self._no_modal():
            self._move_cursor(-event.y)

    def _no_modal(self) -> bool:
        return self.prompt is None and self.picker is None and self.cued is None

    def _on_click(self, pos) -> None:
        if self.cued is not None:
            return
        if self.prompt is not None:
            for rect, action in reversed(self._modal_hits):
                if rect.collidepoint(pos):
                    action()
                    return
            return
        if self.picker is not None:
            for rect, action in reversed(self._modal_hits):
                if rect.collidepoint(pos):
                    action()
                    return
            if self._modal_box is not None and not self._modal_box.collidepoint(pos):
                self.picker = None              # click outside: cancel
            return
        if self.view == "playback" and self._timeline is not None and self._timeline.collidepoint(pos):
            self._dragging = True
            self._seek_to_x(pos[0])
            return
        for rect, action in reversed(self._hits):
            if rect.collidepoint(pos):
                action()
                return

    def _on_key(self, event) -> None:
        k = event.key
        if self.cued is not None:
            self._cued_key(k)
            return
        if self.prompt is not None:
            if self.prompt.get("kind") == "cued" and self._cued_prompt_key(k):
                return
            if k in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self._confirm_prompt()
            elif k == pygame.K_ESCAPE:
                self.prompt = None
            elif k == pygame.K_BACKSPACE:
                self.prompt["text"] = self.prompt["text"][:-1]
            return
        if self.picker is not None:
            if k in (pygame.K_RETURN, pygame.K_KP_ENTER) and self.picker["text"].strip():
                self.apply_picker(self.picker["text"])
            elif k == pygame.K_ESCAPE:
                self.picker = None
            elif k == pygame.K_BACKSPACE:
                self.picker["text"] = self.picker["text"][:-1]
            return
        if k == pygame.K_ESCAPE:
            if self.view == "trials":
                self.set_view("compare")
            else:
                self.running = False
        elif k == pygame.K_TAB:
            order = ["live", "compare"] + (["playback"] if self.playback is not None else []) + \
                    (["trials"] if self.trial_view is not None else [])
            self.set_view(order[(order.index(self.view) + 1) % len(order)])
        elif k == pygame.K_F12:
            path = self.save_screenshot(config.SCREENSHOTS_DIR / f"scope_{datetime.now():%Y%m%d_%H%M%S}.png")
            self.flash(f"Screenshot saved: {path.name}")
        elif self.view == "live":
            self._on_live_key(k)
        elif self.view == "playback":
            self._on_playback_key(k)
        elif self.view == "trials":
            self._on_trials_key(k)
        else:
            self._on_compare_key(k)

    def _on_trace_key(self, k: int) -> bool:
        """Keys LIVE and PLAYBACK share: filter, scale, time span."""
        if k == pygame.K_f:
            self.filtered = not self.filtered
        elif k == pygame.K_UP:
            self.scale_i = max(0, self.scale_i - 1)
        elif k == pygame.K_DOWN:
            self.scale_i = min(len(SCALES_UV) - 1, self.scale_i + 1)
        elif k == pygame.K_LEFTBRACKET:
            self.span_i = max(0, self.span_i - 1)
        elif k == pygame.K_RIGHTBRACKET:
            self.span_i = min(len(SPANS_S) - 1, self.span_i + 1)
        else:
            return False
        return True

    def _on_live_key(self, k: int) -> None:
        if self._on_trace_key(k):
            return
        if k == pygame.K_SPACE:
            self.toggle_recording()
        elif k == pygame.K_t:
            self.open_cued_prompt()
        elif k == pygame.K_r:
            if self.feeds.measure_contact():
                self.flash("Measuring electrode contact (each device pauses for ~5 s)…")
            else:
                self.flash("Nothing streaming yet to measure.", theme.WARN)
        elif k == pygame.K_LEFT:
            self.span_i = max(0, self.span_i - 1)
        elif k == pygame.K_RIGHT:
            self.span_i = min(len(SPANS_S) - 1, self.span_i + 1)

    def _on_playback_key(self, k: int) -> None:
        pb = self.playback
        if self._on_trace_key(k) or pb is None:
            return
        if k == pygame.K_SPACE:
            pb.toggle(self.now)
        elif k == pygame.K_LEFT:
            self.seek(pb.pos - SEEK_S)
        elif k == pygame.K_RIGHT:
            self.seek(pb.pos + SEEK_S)
        elif k == pygame.K_HOME:
            self.seek(0.0)
        elif k == pygame.K_END:
            self.seek(pb.duration)
        elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
            pb.faster(-1)
        elif k in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
            pb.faster(1)
        elif k == pygame.K_t:
            self.open_trials(pb.path)

    def _on_compare_key(self, k: int) -> None:
        if k == pygame.K_UP:
            self._move_cursor(-1)
        elif k == pygame.K_DOWN:
            self._move_cursor(1)
        elif k in (pygame.K_SPACE, pygame.K_RETURN, pygame.K_KP_ENTER):
            if self.library:
                self.toggle_selected(self.library[self.cursor]["path"])
        elif k == pygame.K_LEFT:
            self.metric_i = (self.metric_i - 1) % len(analysis.METRICS)
        elif k == pygame.K_RIGHT:
            self.metric_i = (self.metric_i + 1) % len(analysis.METRICS)
        elif k == pygame.K_c:
            self.selected.clear()
            self._slot.clear()
        elif k == pygame.K_p and self.library:
            self.open_playback(self.library[self.cursor]["path"])
        elif k == pygame.K_t and self.library:
            self.open_trials(self.library[self.cursor]["path"])
        elif k == pygame.K_n and self.library:
            meta = self.library[self.cursor]
            self.prompt = {"title": "Rename recording", "text": meta.get("name", ""), "path": meta["path"],
                           "swallow": ("n", time.monotonic() + 0.25)}

    def _mock_keys(self) -> None:
        if not self.feeds.mock:
            return
        held = pygame.key.get_pressed() if self._no_modal() and self.view == "live" else None
        gaze = next((g for key, g in _MOCK_GAZE_KEYS if held and held[key]), None)
        self.feeds.mock_input(gaze, bool(held and held[pygame.K_j]))

    def set_view(self, view: str) -> None:
        if (view == "playback" and self.playback is None) or (view == "trials" and self.trial_view is None):
            return
        self.view = view
        self._stats_at = -math.inf
        if view == "compare":
            self.refresh_library()

    # -- recording ----------------------------------------------------------------------
    def toggle_recording(self) -> None:
        if self.feeds.recording:
            self.stop_recording()
        else:
            self.feeds.start_recording()
            self.flash("Recording every electrode. Space stops.", theme.BAD)

    def stop_recording(self, ask_name: bool = True) -> Optional[Path]:
        rec = self.feeds.stop_recording()
        run, self.cued = self.cued, None
        if run is not None:
            run.stop()                      # quitting mid-session: keep the trials finished so far
            self.feeds.mock_cue(None)
        if rec is None:
            return None
        if not any(len(d["t"]) for d in rec["devices"].values()):
            self.flash("Nothing recorded: no samples arrived.", theme.WARN)
            return None
        name = ""
        if run is not None:                 # a cued session: its trials and protocol, named up front
            rec["meta"].update(run.meta())
            name, ask_name = run.name, False
        path = recording.save(rec, self.rec_dir, name)
        print(f"[neurover scope] saved {_clock(rec['meta']['duration_s'])} recording to {path}", flush=True)
        self.refresh_library()
        if ask_name:
            self.picker = None
            self.prompt = {"title": f"Saved {_clock(rec['meta']['duration_s'])} recording. Name it?",
                           "text": "", "path": path}
        return path

    def _confirm_prompt(self) -> None:
        p, self.prompt = self.prompt, None
        name = p["text"].strip()
        old = Path(p["path"])
        if not old.exists():
            return
        try:
            meta = recording.info(old)
            if name == meta.get("name", ""):
                return
            new = recording.rename(old, name)
        except Exception as exc:
            self.flash(f"Couldn't rename: {exc}", theme.BAD)
            return
        self._moved(old, new)
        self.flash(f"Saved as {new.name}")

    def _moved(self, old: Path, new: Path) -> None:
        """A recording's file was rewritten (renamed or relabelled): follow it everywhere."""
        if old != new and old in self._slot:
            self._slot[new] = self._slot.pop(old)
            self.selected = [new if s == old else s for s in self.selected]
        self._analysis.pop(old, None)
        if self.playback is not None and self.playback.path == old:
            self._reload_playback(new)
        if self.trial_view is not None and self.trial_view["path"] == old:
            self.open_trials(new, switch=False)
        self.refresh_library()

    # -- electrode positions ------------------------------------------------------------
    def open_picker(self, kind: str, device: str, sdk: str) -> None:
        """Ask where an electrode is. ``kind``: "live" (moves it from now on) or "playback" (corrects
        the recording being played)."""
        if kind == "live" and self.feeds.recording:
            self.flash("Stop recording first: moving an electrode mid-recording would mislabel half of it.",
                       theme.WARN, 4.0)
            return
        src = self.feeds if kind == "live" else self.playback
        if src is None:
            return
        d = next(x for x in src.devices if x.name == device)
        i = d.sdk.index(sdk)
        taken = {}
        for x in src.devices:
            for c, p in zip(x.sdk, x.positions):
                if not (x is d and c == sdk):
                    taken[p] = (f"{x.label} {c}", x is d)
        default = d.config_positions[i] if kind == "live" else None
        leads = [(x.label, x is d, list(zip(x.sdk, x.positions))) for x in src.devices]
        self.picker = {"kind": kind, "device": device, "sdk": sdk, "label": d.label, "current": d.positions[i],
                       "taken": taken, "default": default, "leads": leads, "text": ""}

    def apply_picker(self, position: str) -> bool:
        pk = self.picker
        if pk is None:
            return False
        position = position.strip()
        if position == pk["current"]:
            self.picker = None
            return True
        try:
            if pk["kind"] == "live":
                self.feeds.set_position(pk["device"], pk["sdk"], position)
                if self.placement_path is not None:
                    placement.save(self.feeds.placement(), self.placement_path)
                where = "from now on"
            else:
                old = self.playback.path
                new = recording.relabel(old, pk["device"], pk["sdk"], position)
                self._moved(old, new)
                where = "in this recording"
        except ValueError as exc:
            self.flash(str(exc), theme.BAD, 4.0)
            return False
        except Exception as exc:
            self.flash(f"Couldn't move it: {exc}", theme.BAD, 4.0)
            return False
        swapped = pk["taken"].get(position)
        note = f" (swapped with {swapped[0]})" if swapped and swapped[1] else ""
        self.flash(f"{pk['label']} {pk['sdk']} is at {position} {where}{note}")
        self.picker = None
        self._stats_at = -math.inf
        return True

    # -- playback -----------------------------------------------------------------------
    def open_playback(self, path: Path) -> bool:
        try:
            pb = Playback(recording.load(path))
        except Exception as exc:
            self.flash(f"Couldn't open {Path(path).name}: {exc}", theme.BAD)
            return False
        self.playback = pb
        self.set_view("playback")
        pb.toggle(self.now)                 # start playing
        return True

    def _reload_playback(self, path: Path) -> None:
        old = self.playback
        pb = Playback(recording.load(path))
        pb.pos, pb.speed_i, pb.playing, pb._last = min(old.pos, pb.duration), old.speed_i, old.playing, old._last
        self.playback = pb

    def seek(self, pos: float) -> None:
        if self.playback is not None:
            self.playback.seek(pos)
            self._stats_at = -math.inf

    def _seek_to_x(self, x: float) -> None:
        r = self._timeline
        if r is not None and self.playback is not None and r.w > 0:
            self.seek((x - r.x) / r.w * self.playback.duration)

    # -- compare state ------------------------------------------------------------------
    def refresh_library(self) -> None:
        focused = self.library[self.cursor]["path"] if self.library else None
        self.library = recording.list_recordings(self.rec_dir)
        paths = [m["path"] for m in self.library]
        self.selected = [p for p in self.selected if p in paths]
        self._slot = {p: s for p, s in self._slot.items() if p in paths}
        self.cursor = paths.index(focused) if focused in paths else 0

    def _move_cursor(self, step: int) -> None:
        if self.library:
            self.cursor = max(0, min(len(self.library) - 1, self.cursor + step))

    def toggle_selected(self, path: Path) -> None:
        if path in self.selected:
            self.selected.remove(path)
            self._slot.pop(path, None)
            return
        if len(self.selected) >= len(REC_COLORS):
            self.flash(f"At most {len(REC_COLORS)} recordings at once (Space on one to drop it).", theme.WARN)
            return
        used = set(self._slot.values())
        self._slot[path] = next(i for i in range(len(REC_COLORS)) if i not in used)
        self.selected.append(path)

    def shown(self) -> list[tuple[dict, tuple]]:
        """(recording meta, colour) for what the compare view shows: the selection, else the focused one."""
        by_path = {m["path"]: m for m in self.library}
        if self.selected:
            return [(by_path[p], REC_COLORS[self._slot[p]]) for p in self.selected if p in by_path]
        return [(self.library[self.cursor], REC_COLORS[0])] if self.library else []

    def stats_for(self, meta: dict) -> dict:
        path = meta["path"]
        cached = self._analysis.get(path)
        if cached is None or cached[0] != meta["mtime"]:
            try:
                stats = analysis.analyse_recording(recording.load(path))
            except Exception as exc:
                print(f"[neurover scope] couldn't analyse {path.name}: {exc}", flush=True)
                stats = {}
            cached = (meta["mtime"], stats)
            self._analysis[path] = cached
        return cached[1]

    # -- per-row numbers ----------------------------------------------------------------
    def _rows_source(self):
        """(what the rows show, the time their right edge is at) for LIVE / PLAYBACK."""
        if self.view == "playback" and self.playback is not None:
            return self.playback, self.playback.head
        return self.feeds, self.now

    def _update_stats(self) -> None:
        if self.view in ("compare", "trials") or self.cued is not None or self.now - self._stats_at < STATS_EVERY_S:
            return
        self._stats_at = self.now
        src, t_end = self._rows_source()
        out = {}
        for d in src.devices:
            silent = d.silent(t_end)
            t, x = d.window(t_end, STATS_S, "raw")
            for col, sdk in enumerate(d.sdk):
                out[(d.name, col)] = {} if silent or len(t) < 2 else analysis.analyse(
                    t, x[:, col], d.fs, resistance=d.resistance.get(sdk), nperseg_s=2.0)
        self.row_stats[self.view] = out

    # -- drawing ------------------------------------------------------------------------
    def flash(self, text: str, color=theme.TEXT, seconds: float = 3.0) -> None:
        self.toast = (text, time.monotonic() + seconds, color)

    def save_screenshot(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(self.screen, str(path))
        return path

    def draw(self) -> None:
        surf = self.screen
        w, h = surf.get_size()
        f = self.fonts
        f.set_scale(min(w / DESIGN[0], h / DESIGN[1]))
        self._hits = []
        self._modal_hits = []
        self._modal_box = None
        self._timeline = None
        self._mouse = pygame.mouse.get_pos()
        self._want_hand = False
        if self.cued is not None:           # a cued session owns the whole window
            self._draw_cued(surf)
            self._set_cursor()
            return
        surf.fill(theme.BG)
        head = pygame.Rect(0, 0, w, f.px(64))
        foot = pygame.Rect(0, h - f.px(34), w, f.px(34))
        body = pygame.Rect(f.px(12), head.bottom + f.px(6), w - f.px(24), foot.top - head.bottom - f.px(10))
        self._draw_header(surf, head)
        if self.view == "live":
            self._draw_rows(surf, body, "live")
        elif self.view == "playback" and self.playback is not None:
            bar = pygame.Rect(body.x, body.bottom - f.px(78), body.w, f.px(78))
            self._draw_rows(surf, pygame.Rect(body.x, body.y, body.w, bar.y - body.y - f.px(6)), "playback")
            self._draw_transport(surf, bar)
        elif self.view == "trials":
            self._draw_trials(surf, body)
        else:
            self._draw_compare(surf, body)
        self._draw_footer(surf, foot)
        if self.prompt is not None:
            if self.prompt.get("kind") == "cued":
                self._draw_cued_prompt(surf)
            else:
                self._draw_prompt(surf)
        elif self.picker is not None:
            self._draw_picker(surf)
        if self.toast is not None:
            text, until, col = self.toast
            if time.monotonic() > until:
                self.toast = None
            else:
                pos = (w / 2, foot.top - f.px(16) - (f.px(84) if self.view == "playback" else 0))
                r = f.render(text, 20, col, bold=True).get_rect(midbottom=(round(pos[0]), round(pos[1])))
                theme.panel(surf, r.inflate(f.px(28), f.px(14)), theme.PANEL_HI, theme.EDGE, f.px(8))
                f.text(surf, text, 20, col, pos, "midbottom", bold=True)
        self._set_cursor()

    def _hover(self, rect: pygame.Rect) -> bool:
        over = rect.collidepoint(self._mouse)
        self._want_hand |= over
        return over

    def _set_cursor(self) -> None:
        if self._want_hand == self._hand or self._headless:
            return
        self._hand = self._want_hand
        try:
            pygame.mouse.set_system_cursor(pygame.SYSTEM_CURSOR_HAND if self._hand else pygame.SYSTEM_CURSOR_ARROW)
        except Exception:
            pass

    def _draw_header(self, surf, r: pygame.Rect) -> None:
        f = self.fonts
        pygame.draw.rect(surf, theme.PANEL, r)
        pygame.draw.line(surf, theme.EDGE, r.bottomleft, r.bottomright, 1)
        x = r.x + f.px(16)
        title = f.text(surf, "NEUROVER SCOPE", 24, theme.TEXT, (x, r.centery), "midleft", bold=True)
        x = title.right + f.px(22)
        tabs = [("live", "LIVE"), ("compare", "COMPARE")] + ([("playback", "PLAYBACK")] if self.playback else []) + \
               ([("trials", "TRIALS")] if self.trial_view else [])
        for view, label in tabs:
            on = self.view == view
            txt = f.render(label, 17, theme.INK if on else theme.MUTED, bold=True)
            tab = pygame.Rect(x, r.centery - f.px(16), txt.get_width() + f.px(26), f.px(32))
            theme.panel(surf, tab, theme.TEXT if on else theme.PANEL_HI, None if on else theme.EDGE, f.px(16))
            surf.blit(txt, txt.get_rect(center=tab.center))
            self._hover(tab)
            self._hits.append((tab, lambda v=view: self.set_view(v)))
            x = tab.right + f.px(8)
        x += f.px(14)
        if self.feeds.fatal:
            f.text(surf, self.feeds.fatal, 17, theme.BAD, (x, r.centery), "midleft", bold=True)
        else:
            for d in self.feeds.devices:
                silent = d.state == "streaming" and d.silent(self.now)
                col = theme.WARN if silent else _state_color(d.state)
                theme.dot(surf, col, (x + f.px(6), r.centery), f.px(6))
                state = "silent" if silent else d.state
                bat = f" {d.battery}%" if d.battery is not None else ""
                rr = f.text(surf, f"{d.label}", 18, theme.TEXT, (x + f.px(18), r.centery), "midleft", bold=True)
                rr = f.text(surf, f"{state}{bat}", 16, col, (rr.right + f.px(6), r.centery), "midleft")
                x = rr.right + f.px(22)
        # record control, right
        rx = r.right - f.px(16)
        if self.feeds.recording:
            el = self.feeds.recorder.elapsed(self.now)
            pulse = 0.5 + 0.5 * math.sin(self.now * 6)
            btn = pygame.Rect(0, 0, f.px(250), f.px(38))
            btn.midright = (rx, r.centery)
            theme.panel(surf, btn, theme.lerp_color(theme.PANEL_HI, (90, 24, 32), pulse), theme.BAD, f.px(19), 2)
            theme.dot(surf, theme.BAD, (btn.x + f.px(20), btn.centery), f.px(7))
            f.text(surf, f"REC  {_clock(el)}", 20, theme.TEXT, (btn.x + f.px(36), btn.centery), "midleft",
                   bold=True)
            f.text(surf, "Space stop" if self.view == "live" else "click stop", 14, theme.MUTED,
                   (btn.right - f.px(14), btn.centery), "midright")
        else:
            btn = pygame.Rect(0, 0, f.px(170), f.px(38))
            btn.midright = (rx, r.centery)
            theme.panel(surf, btn, theme.PANEL_HI, theme.EDGE, f.px(19))
            theme.dot(surf, theme.BAD, (btn.x + f.px(20), btn.centery), f.px(7))
            f.text(surf, "Record", 19, theme.TEXT, (btn.x + f.px(36), btn.centery), "midleft", bold=True)
            if self.view == "live":
                f.text(surf, "Space", 14, theme.MUTED, (btn.right - f.px(14), btn.centery), "midright")
        self._hover(btn)
        self._hits.append((btn, self.toggle_recording))

    def _draw_footer(self, surf, r: pygame.Rect) -> None:
        f = self.fonts
        trace = [f"F trace: {'1–40 Hz + notch' if self.filtered else 'raw − DC'}",
                 f"↑↓ ±{SCALES_UV[self.scale_i]:g} µV", f"[ ] {SPANS_S[self.span_i]:g} s"]
        if self.view == "live":
            parts = trace + ["click a name to move it", "R contact", "Space record", "T cued trials", "Tab compare",
                             "Esc quit"]
            if self.feeds.mock:
                parts.append("mock: hold 1–4 stare at box · J jaw")
        elif self.view == "playback":
            parts = ["Space play/pause", f"←→ ±{SEEK_S:g} s", "Home/End", "− = speed"] + trace + \
                    ["click a name to fix it in this recording"] + \
                    (["T trial analysis"] if self.playback and self.playback.trials else []) + ["Tab", "Esc quit"]
        elif self.view == "trials":
            parts = ["A reference: device average / as recorded", "←→ or click: electrode for the time course",
                     "P play it back", "hover a bar for its numbers", "Tab", "Esc back"]
        else:
            parts = ["↑↓ recording", "Space select (up to 6)", "P play", "T trial analysis", "←→ metric", "N rename",
                     "C clear", "Tab", "Esc quit"]
        f.text(surf, "   ·   ".join(parts), 15, theme.DIM, (r.x + f.px(16), r.centery), "midleft",
               max_width=r.w - f.px(32))

    # -- LIVE / PLAYBACK rows ------------------------------------------------------------------
    def _draw_rows(self, surf, body: pygame.Rect, kind: str) -> None:
        f = self.fonts
        src, t_end = self._rows_source()
        stats = self.row_stats.get(kind, {})
        devices = src.devices
        group_h = f.px(30)
        row_h = max(f.px(30), (body.h - group_h * len(devices)) / max(1, len(src.nodes)))
        label_w, spec_w, stat_w = f.px(176), f.px(300), f.px(230)
        gap = f.px(10)
        trace_x = body.x + label_w
        trace_w = body.w - label_w - spec_w - stat_w - 3 * gap
        spec_x = trace_x + trace_w + gap
        stat_x = spec_x + spec_w + gap
        span, scale = SPANS_S[self.span_i], SCALES_UV[self.scale_i]

        curves = {key: (st["freqs"], st["psd"]) for key, st in stats.items() if st.get("psd") is not None}
        top_db = self._top_db(curves.values())

        y = body.y
        for d in devices:
            self._draw_group_header(surf, pygame.Rect(body.x, y, body.w, group_h), d, t_end, kind)
            y += group_h
            nodes = [nd for nd in src.nodes if nd.device == d.name]
            t_all, x_all = d.window(t_end, span + 0.5, "display" if self.filtered else "raw")
            for nd in nodes:
                row = pygame.Rect(body.x, round(y), body.w, round(row_h) - f.px(4))
                st = stats.get((d.name, nd.col), {})
                status, sev = analysis.status(st)
                theme.panel(surf, row, theme.PANEL, None, f.px(6))
                lr = pygame.Rect(row.x, row.y, label_w, row.h)
                self._draw_node_label(surf, lr, nd.position, f"{d.label} {nd.sdk}", d.resistance.get(nd.sdk),
                                      SEVERITY[sev], kind)
                self._hits.append((lr, lambda k=kind, dv=d.name, c=nd.sdk: self.open_picker(k, dv, c)))
                tr = pygame.Rect(trace_x, row.y + f.px(3), trace_w, row.h - f.px(6))
                self._draw_time_grid(surf, tr, span)
                if len(t_all) > 1:
                    yv = x_all[:, nd.col]
                    if not self.filtered:
                        recent = t_all >= t_all[-1] - span
                        yv = yv - (np.median(yv[recent]) if recent.any() else 0.0)
                    self._draw_trace(surf, tr, t_all, yv, t_end, span, scale,
                                     theme.BAD if sev == "bad" else theme.TRAIL)
                sr = pygame.Rect(spec_x, row.y + f.px(3), spec_w, row.h - f.px(6))
                psd = curves.get((d.name, nd.col))
                self._draw_psd(surf, sr, [(psd[0], psd[1], theme.TEXT)] if psd else [], top_db,
                               axis=nd is nodes[-1] and d is devices[-1])
                self._draw_row_stats(surf, pygame.Rect(stat_x, row.y, stat_w, row.h), st, status, sev)
                y += row_h

    def _draw_group_header(self, surf, r: pygame.Rect, d, t_end: float, kind: str) -> None:
        f = self.fonts
        silent = d.state == "streaming" and d.silent(t_end)
        col = theme.WARN if silent else _state_color(d.state)
        theme.dot(surf, col, (r.x + f.px(8), r.centery), f.px(5))
        x = f.text(surf, d.label.upper(), 17, theme.TEXT, (r.x + f.px(20), r.centery), "midleft", bold=True).right
        bits = []
        if d.info.get("kind"):
            bits.append(d.info["kind"])
        serial = d.info.get("serial") or d.serial
        if serial:
            bits.append(f"serial {serial}")
        if d.battery is not None:
            bits.append(f"battery {d.battery}%")
        rate = d.rate(t_end)
        if rate and d.state == "streaming" and not silent:
            bits.append(f"{rate:.0f} Hz")
        if bits:
            x = f.text(surf, "  ·  ".join(bits), 15, theme.MUTED, (x + f.px(12), r.centery), "midleft").right
        if silent:
            msg = "no data here (Bluetooth gap)" if kind == "playback" else "no samples for over a second"
        else:
            msg = d.message if d.state != "streaming" else ""
        if msg:
            f.text(surf, msg, 15, col, (x + f.px(16), r.centery), "midleft", max_width=r.right - x - f.px(16))

    def _draw_node_label(self, surf, r: pygame.Rect, position: str, sub: str, ohm: Optional[float], sev_col,
                         kind: str) -> None:
        f = self.fonts
        hover = self._hover(r) and self._no_modal()
        if hover:
            theme.panel(surf, r.inflate(-f.px(4), -f.px(4)), theme.PANEL_HI, theme.TEXT, f.px(6))
        theme.dot(surf, sev_col, (r.x + f.px(12), r.centery), f.px(6))
        x = r.x + f.px(26)
        f.text(surf, position, 22, theme.TEXT, (x, r.centery + f.px(2)), "bottomleft", bold=True)
        if hover:
            line, col = ("click to move" if kind == "live" else "click to fix"), theme.TEXT
        else:
            line, col = sub, theme.DIM
            if ohm is not None:
                line += "  " + _ohm_text(ohm)
                col = theme.GOOD if math.isfinite(ohm) and ohm < config.EEG_CONTACT_GOOD_OHM else theme.BAD
        f.text(surf, line, 13, col, (x, r.centery + f.px(2)), "topleft", max_width=r.right - x - f.px(4))

    def _draw_time_grid(self, surf, r: pygame.Rect, span: float) -> None:
        step = 1.0 if span <= 10 else 5.0
        k = 1
        while k * step < span:
            x = r.right - k * step / span * r.w
            pygame.draw.line(surf, theme.PANEL_HI, (x, r.y), (x, r.bottom), 1)
            k += 1
        pygame.draw.line(surf, theme.PANEL_HI, (r.x, r.centery), (r.right, r.centery), 1)

    def _draw_trace(self, surf, r: pygame.Rect, t: np.ndarray, y: np.ndarray, t_end: float, span: float,
                    scale_uv: float, color) -> None:
        t0 = t_end - span
        sel = (t > t0) & (t <= t_end + 0.5)
        t, y = t[sel], y[sel]
        if len(t) < 2:
            return
        half = r.h * 0.48
        px = np.clip(r.x + (t - t0) / span * r.w, r.x, r.right)
        py = r.centery - np.clip(y / scale_uv, -1.0, 1.0) * half
        for a, b in analysis.runs(t):
            if b - a < 2:
                continue
            xs, ys = px[a:b], py[a:b]
            if b - a > 2 * r.w:   # several samples per pixel: draw each column's min..max (keeps spikes)
                cols = xs.astype(int)
                starts = np.flatnonzero(np.r_[True, np.diff(cols) != 0])
                pts = np.empty((2 * len(starts), 2))
                pts[0::2, 0] = pts[1::2, 0] = xs[starts]
                pts[0::2, 1] = np.minimum.reduceat(ys, starts)
                pts[1::2, 1] = np.maximum.reduceat(ys, starts)
            else:
                pts = np.column_stack([xs, ys])
            if len(pts) >= 2:
                pygame.draw.lines(surf, color, False, pts.tolist(), 1)

    def _draw_row_stats(self, surf, r: pygame.Rect, st: dict, status: str, sev: str) -> None:
        f = self.fonts
        col = SEVERITY[sev]
        pill = f.render(status, 13, theme.INK if sev != "none" else theme.MUTED, bold=True)
        pr = pygame.Rect(0, 0, pill.get_width() + f.px(14), f.px(20))
        pr.topright = (r.right - f.px(8), r.y + f.px(6))
        theme.panel(surf, pr, col if sev != "none" else theme.PANEL_HI, None, f.px(10))
        surf.blit(pill, pill.get_rect(center=pr.center))
        M = analysis.METRIC_BY_KEY
        rows = [("RMS", "rms"), ("peak", "peak"), (f"{config.EEG_NOTCH_HZ:g} Hz", "line"), ("DC", "dc")]
        lh = min(f.px(18), (r.h - f.px(8)) / 4)
        y0 = r.y + max(f.px(4), (r.h - 4 * lh) / 2)
        for i, (name, key) in enumerate(rows):
            m = M[key]
            y = y0 + i * lh
            f.text(surf, name, 14, theme.DIM, (r.x + f.px(8), y), "topleft")
            f.text(surf, f"{analysis.format_value(m, st.get(key))} {m.unit}", 14, theme.TEXT,
                   (r.x + f.px(64), y), "topleft", mono=True)

    # -- playback transport -------------------------------------------------------------------
    def _draw_transport(self, surf, r: pygame.Rect) -> None:
        f = self.fonts
        pb = self.playback
        theme.panel(surf, r, theme.PANEL, theme.EDGE, f.px(10))
        # play / pause button
        btn_c = (r.x + f.px(34), r.y + f.px(26))
        btn = pygame.Rect(0, 0, f.px(40), f.px(40))
        btn.center = btn_c
        hover = self._hover(btn)
        pygame.draw.circle(surf, theme.TEXT if hover else theme.PANEL_HI, btn_c, f.px(20))
        ink = theme.INK if hover else theme.TEXT
        if pb.playing:
            for dx in (-f.px(6), f.px(3)):
                pygame.draw.rect(surf, ink, (btn_c[0] + dx, btn_c[1] - f.px(8), f.px(4), f.px(16)))
        else:
            s = f.px(8)
            pygame.draw.polygon(surf, ink, [(btn_c[0] - s * 0.6, btn_c[1] - s), (btn_c[0] - s * 0.6, btn_c[1] + s),
                                            (btn_c[0] + s, btn_c[1])])
        self._hits.append((btn, lambda: pb.toggle(self.now)))
        x = btn.right + f.px(14)
        x = f.text(surf, f"{_clock_tenths(pb.pos)} / {_clock_tenths(pb.duration)}", 18, theme.TEXT,
                   (x, btn_c[1]), "midleft", mono=True).right + f.px(14)
        # speed chips
        for i, sp in enumerate(SPEEDS):
            on = i == pb.speed_i
            txt = f.render(f"{sp:g}×", 14, theme.INK if on else theme.MUTED, bold=on)
            chip = pygame.Rect(x, btn_c[1] - f.px(12), txt.get_width() + f.px(14), f.px(24))
            theme.panel(surf, chip, theme.TEXT if on else theme.PANEL_HI, None, f.px(12))
            surf.blit(txt, txt.get_rect(center=chip.center))
            self._hover(chip)
            self._hits.append((chip, lambda i=i: setattr(pb, "speed_i", i)))
            x = chip.right + f.px(4)
        cue = pb.cue_at(pb.head)
        if cue is not None:                 # which cue the subject was looking at, here
            n, tr = cue
            theme.dot(surf, class_color(tr["label"]), (x + f.px(22), btn_c[1]), f.px(7))
            x = f.text(surf, f"{'practice' if n == 0 else f'trial {n}'}: {CLASS_TEXT.get(tr['label'], tr['label'])}",
                       16, theme.TEXT, (x + f.px(34), btn_c[1]), "midleft", bold=True).right
        created = datetime.fromtimestamp(pb.meta.get("created", 0))
        f.text(surf, f"{pb.name}  ·  {created:%b %d %H:%M}" + ("  ·  mock" if pb.meta.get("mock") else ""),
               16, theme.MUTED, (r.right - f.px(16), btn_c[1]), "midright", bold=True,
               max_width=r.right - x - f.px(40))
        # timeline: progress, playhead, logged events above, each device's coverage below
        tl = pygame.Rect(r.x + f.px(16), r.y + f.px(52), r.w - f.px(32), f.px(8))
        self._timeline = tl.inflate(0, f.px(22))
        self._hover(self._timeline)
        dur = max(pb.duration, 1e-9)

        def tx(t_abs: float) -> float:
            return tl.x + (t_abs - pb.t0) / dur * tl.w

        pygame.draw.rect(surf, theme.PANEL_HI, tl, border_radius=f.px(4))
        done = tl.copy()
        done.w = round(tl.w * pb.pos / dur)
        pygame.draw.rect(surf, theme.TRAIL, done, border_radius=f.px(4))
        lane_h = f.px(3)
        for k, d in enumerate(pb.devices):
            ly = tl.bottom + f.px(3) + k * (lane_h + f.px(2))
            pygame.draw.rect(surf, (70, 30, 38), (tl.x, ly, tl.w, lane_h))
            for a, b in d.coverage:
                pygame.draw.rect(surf, theme.GOOD, (tx(a), ly, max(1, tx(b) - tx(a)), lane_h))
        near = None
        for t_rel, dev, text in pb.events:
            ex = tl.x + t_rel / dur * tl.w
            col = theme.WARN if text.startswith("resist") else theme.BAD if "lost" in text or "error" in text \
                else theme.MUTED
            pygame.draw.line(surf, col, (ex, tl.y - f.px(6)), (ex, tl.y - 1), 2)
            if abs(self._mouse[0] - ex) < f.px(5) and abs(self._mouse[1] - tl.y) < f.px(16):
                near = (ex, f"{_clock_tenths(t_rel)}  {dev}: {text}")
        n = 0
        for tr in pb.trials:                # cue onsets, coloured by class, and how long each cue showed
            n += 0 if tr.get("practice") else 1
            col = class_color(tr["label"])
            if tr.get("practice"):
                col = theme.lerp_color(theme.PANEL_HI, col, 0.5)
            x0, x1 = tx(tr["cue"]), tx(tr.get("end", tr["cue"]))
            pygame.draw.rect(surf, col, (x0, tl.y - f.px(5), max(1, x1 - x0), f.px(3)))
            pygame.draw.line(surf, col, (x0, tl.y - f.px(13)), (x0, tl.y - 1), 2)
            if abs(self._mouse[0] - x0) < f.px(5) and abs(self._mouse[1] - tl.y) < f.px(16):
                near = (x0, f"{_clock_tenths(tr['cue'] - pb.t0)}  {'practice' if tr.get('practice') else f'trial {n}'}"
                            f": {CLASS_TEXT.get(tr['label'], tr['label'])} cue")
        hx = tl.x + tl.w * pb.pos / dur
        pygame.draw.circle(surf, theme.TEXT, (round(hx), tl.centery), f.px(8))
        if near is not None:
            f.text(surf, near[1], 14, theme.TEXT, (near[0], tl.y - f.px(8)), "midbottom",
                   max_width=r.w - f.px(40))

    # -- spectra ----------------------------------------------------------------------------
    @staticmethod
    def _top_db(curves) -> float:
        lo, hi = analysis.SPECTRUM_HZ
        tops = []
        for freqs, psd in curves:
            if freqs is None:
                continue
            sel = (freqs >= lo) & (freqs <= hi)
            if sel.any():
                tops.append(10 * math.log10(float(np.max(psd[sel])) + 1e-12))
        return (max(tops) + 3.0) if tops else 20.0

    def _draw_psd(self, surf, r: pygame.Rect, curves: list, top_db: float, axis: bool = False) -> None:
        f = self.fonts
        lo, hi = analysis.SPECTRUM_HZ
        pygame.draw.rect(surf, theme.PANEL_HI, r, border_radius=f.px(4))

        def fx(freq: float) -> float:
            return r.x + (freq - lo) / (hi - lo) * r.w

        a0, a1 = config.ALPHA_BAND
        pygame.draw.rect(surf, (40, 46, 60), (fx(a0), r.y, fx(a1) - fx(a0), r.h))
        for label in analysis.TARGET_ORDER:
            x = fx(analysis.TARGETS[label])
            pygame.draw.line(surf, (70, 78, 96), (x, r.y), (x, r.bottom), 1)
        if axis:
            for tick in (10, 20, 30, 40):
                f.text(surf, f"{tick}", 12, theme.DIM, (fx(tick), r.bottom - f.px(1)), "midbottom")
        for freqs, psd, color in curves:
            if freqs is None:
                continue
            sel = (freqs >= lo) & (freqs <= hi)
            if sel.sum() < 2:
                continue
            db = 10 * np.log10(psd[sel] + 1e-12)
            ys = r.bottom - np.clip((db - (top_db - PSD_SPAN_DB)) / PSD_SPAN_DB, 0.0, 1.0) * r.h
            xs = r.x + (freqs[sel] - lo) / (hi - lo) * r.w
            pygame.draw.lines(surf, color, False, np.column_stack([xs, ys]).tolist(), 2)

    # -- compare view -----------------------------------------------------------------------
    def _draw_compare(self, surf, body: pygame.Rect) -> None:
        f = self.fonts
        list_w = f.px(400)
        lr = pygame.Rect(body.x, body.y, list_w, body.h)
        rr = pygame.Rect(lr.right + f.px(12), body.y, body.w - list_w - f.px(12), body.h)
        self._draw_library(surf, lr)
        self._draw_comparison(surf, rr)

    def _draw_library(self, surf, r: pygame.Rect) -> None:
        f = self.fonts
        theme.panel(surf, r, theme.PANEL, theme.EDGE, f.px(10))
        f.text(surf, f"RECORDINGS ({len(self.library)})", 16, theme.MUTED, (r.x + f.px(14), r.y + f.px(12)), bold=True)
        f.text(surf, str(self.rec_dir.name) + "/", 14, theme.DIM, (r.right - f.px(14), r.y + f.px(13)), "topright")
        if not self.library:
            for i, line in enumerate(("No recordings yet.", "Tab to LIVE, then Space to record.")):
                f.text(surf, line, 17, theme.DIM, (r.centerx, r.y + f.px(80) + i * f.px(26)), "center")
            return
        row_h = f.px(62)
        top = r.y + f.px(40)
        visible = max(1, (r.bottom - top - f.px(8)) // row_h)
        if self.cursor < self._list_top:
            self._list_top = self.cursor
        elif self.cursor >= self._list_top + visible:
            self._list_top = self.cursor - visible + 1
        self._list_top = max(0, min(self._list_top, max(0, len(self.library) - visible)))
        for j, meta in enumerate(self.library[self._list_top:self._list_top + visible]):
            i = self._list_top + j
            row = pygame.Rect(r.x + f.px(8), top + j * row_h, r.w - f.px(16), row_h - f.px(6))
            path = meta["path"]
            sel = path in self._slot
            focus = i == self.cursor
            theme.panel(surf, row, theme.PANEL_HI if focus else theme.PANEL, theme.TEXT if focus else None, f.px(8))
            sw = pygame.Rect(row.x + f.px(10), row.y + f.px(10), f.px(14), row.h - f.px(20))
            if sel:
                pygame.draw.rect(surf, REC_COLORS[self._slot[path]], sw, border_radius=f.px(4))
            else:
                pygame.draw.rect(surf, theme.EDGE, sw, width=1, border_radius=f.px(4))
            # play button, right
            pc = (row.right - f.px(26), row.centery)
            play = pygame.Rect(0, 0, f.px(36), f.px(36))
            play.center = pc
            ph = self._hover(play)
            pygame.draw.circle(surf, theme.TEXT if ph else theme.PANEL_HI if focus else theme.BG, pc, f.px(17))
            s = f.px(7)
            pygame.draw.polygon(surf, theme.INK if ph else theme.TEXT,
                                [(pc[0] - s * 0.55, pc[1] - s), (pc[0] - s * 0.55, pc[1] + s), (pc[0] + s, pc[1])])
            x = sw.right + f.px(12)
            right = play.x - f.px(10)
            created = datetime.fromtimestamp(meta.get("created", 0))
            name = meta.get("name") or "untitled"
            f.text(surf, name, 18, theme.TEXT if meta.get("name") else theme.MUTED, (x, row.y + f.px(8)),
                   bold=True, max_width=right - x - f.px(56))
            f.text(surf, _clock(meta.get("duration_s", 0)), 16, theme.TEXT, (right, row.y + f.px(9)),
                   "topright", mono=True)
            labels = " + ".join(d.get("label", d["name"]) for d in meta.get("devices", []))
            sub = f"{created:%b %d %H:%M:%S}  ·  {labels}" + ("  ·  mock" if meta.get("mock") else "")
            n_trials = sum(1 for t in meta.get("trials") or [] if not t.get("practice"))
            if n_trials:                    # a cued session: its analysis is one click away
                txt = f.render(f"{n_trials} trials", 13, theme.TEXT, bold=True)
                chip = pygame.Rect(0, 0, txt.get_width() + f.px(16), f.px(20))
                chip.topright = (right, row.y + f.px(31))
                hov = self._hover(chip)
                theme.panel(surf, chip, theme.EDGE if hov else theme.PANEL_HI, theme.EDGE, f.px(10))
                surf.blit(txt, txt.get_rect(center=chip.center))
                trial_hit = (chip, lambda i=i, p=path: self._trials_row(i, p))
                right = chip.x - f.px(8)
            f.text(surf, sub, 14, theme.DIM, (x, row.y + f.px(32)), max_width=right - x)
            self._hover(row)
            self._hits.append((row, lambda i=i, p=path: self._click_row(i, p)))
            self._hits.append((play, lambda i=i, p=path: self._play_row(i, p)))
            if n_trials:
                self._hits.append(trial_hit)
        if len(self.library) > visible:
            f.text(surf, f"{self._list_top + 1}–{min(len(self.library), self._list_top + visible)} of "
                         f"{len(self.library)}", 13, theme.DIM, (r.centerx, r.bottom - f.px(4)), "midbottom")

    def _click_row(self, i: int, path: Path) -> None:
        self.cursor = i
        self.toggle_selected(path)

    def _play_row(self, i: int, path: Path) -> None:
        self.cursor = i
        self.open_playback(path)

    def _trials_row(self, i: int, path: Path) -> None:
        self.cursor = i
        self.open_trials(path)

    def _draw_comparison(self, surf, r: pygame.Rect) -> None:
        f = self.fonts
        metric = analysis.METRICS[self.metric_i]
        # metric chips
        x, y = r.x, r.y
        chip_h = f.px(28)
        for i, m in enumerate(analysis.METRICS):
            on = i == self.metric_i
            txt = f.render(m.label, 14, theme.INK if on else theme.MUTED, bold=on)
            chip = pygame.Rect(x, y, txt.get_width() + f.px(18), chip_h)
            if chip.right > r.right:
                x, y = r.x, y + chip_h + f.px(6)
                chip.topleft = (x, y)
            theme.panel(surf, chip, theme.TEXT if on else theme.PANEL, None if on else theme.EDGE, f.px(14))
            surf.blit(txt, txt.get_rect(center=chip.center))
            self._hover(chip)
            self._hits.append((chip, lambda i=i: setattr(self, "metric_i", i)))
            x = chip.right + f.px(6)
        y += chip_h + f.px(8)
        f.text(surf, f"{metric.label} ({metric.unit}): {metric.help}", 14, theme.DIM, (r.x + f.px(2), y),
               max_width=r.w)
        y += f.px(26)
        shown = self.shown()
        legend_h = f.px(30)
        grid = pygame.Rect(r.x, y, r.w, r.bottom - y - legend_h)
        if not shown:
            f.text(surf, "Pick recordings on the left (Space) to compare them.", 20, theme.DIM, grid.center, "center")
            return
        stats = [(meta, col, self.stats_for(meta)) for meta, col in shown]
        positions = self._union_positions([s for _, _, s in stats])
        if not positions:
            f.text(surf, "These recordings have no electrode data.", 20, theme.DIM, grid.center, "center")
            return
        self._draw_legend(surf, pygame.Rect(r.x, grid.bottom + f.px(6), r.w, legend_h - f.px(6)), stats)
        cols = 4 if len(positions) > 4 else len(positions)
        nrows = math.ceil(len(positions) / cols)
        gap = f.px(10)
        pw = (grid.w - gap * (cols - 1)) / cols
        ph = (grid.h - gap * (nrows - 1)) / nrows
        top_db = self._top_db((s[p]["freqs"], s[p]["psd"]) for _, _, s in stats for p in positions
                              if p in s and s[p].get("psd") is not None)
        values = [s[p].get(metric.key) for _, _, s in stats for p in positions if p in s]
        finite = [abs(v) for v in values if v is not None and math.isfinite(v)]
        vmax = max(finite) if finite else 1.0
        if metric.key.startswith("snr_"):
            vmax = max(vmax, 2.0)
        for k, pos in enumerate(positions):
            pr = pygame.Rect(round(grid.x + (k % cols) * (pw + gap)), round(grid.y + (k // cols) * (ph + gap)),
                             round(pw), round(ph))
            self._draw_node_panel(surf, pr, pos, stats, metric, top_db, vmax)

    @staticmethod
    def _union_positions(all_stats: list[dict]) -> list[str]:
        rank = {p: i for i, p in enumerate(config.EEG_DISPLAY_ORDER)}
        seen = []
        for s in all_stats:
            for p in s:
                if p not in seen:
                    seen.append(p)
        return sorted(seen, key=lambda p: (rank.get(p, len(rank)), seen.index(p)))

    def _draw_node_panel(self, surf, r: pygame.Rect, pos: str, stats, metric, top_db: float, vmax: float) -> None:
        f = self.fonts
        theme.panel(surf, r, theme.PANEL, theme.EDGE, f.px(8))
        pad = f.px(10)
        f.text(surf, pos, 20, theme.TEXT, (r.x + pad, r.y + f.px(6)), bold=True)
        where = sorted({f"{s[pos]['label']} {s[pos]['sdk']}" for _, _, s in stats if pos in s})
        f.text(surf, " / ".join(where), 13, theme.DIM, (r.right - pad, r.y + f.px(10)), "topright",
               max_width=r.w - f.px(70))
        bar_h = f.px(17)
        bars_h = len(stats) * bar_h + f.px(6)
        spec = pygame.Rect(r.x + pad, r.y + f.px(32), r.w - 2 * pad, r.h - f.px(32) - bars_h - pad)
        curves = [(s[pos]["freqs"], s[pos]["psd"], col) for _, col, s in stats
                  if pos in s and s[pos].get("psd") is not None]
        if spec.h > f.px(30):
            self._draw_psd(surf, spec, curves, top_db, axis=True)
            if not curves:
                f.text(surf, "no data", 15, theme.DIM, spec.center, "center")
        y = spec.bottom + f.px(6) if spec.h > f.px(30) else r.y + f.px(32)
        label_w = f.px(62)
        track = pygame.Rect(r.x + pad, 0, r.w - 2 * pad - label_w, bar_h - f.px(5))
        for meta, col, s in stats:
            v = s.get(pos, {}).get(metric.key)
            track.y = round(y)
            pygame.draw.rect(surf, theme.PANEL_HI, track, border_radius=f.px(3))
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                frac = 1.0 if math.isinf(v) else min(1.0, abs(v) / vmax) if vmax > 0 else 0.0
                fill = track.copy()
                fill.w = max(f.px(2), round(track.w * frac))
                pygame.draw.rect(surf, theme.BAD if math.isinf(v) else col, fill, border_radius=f.px(3))
            if metric.key.startswith("snr_") and vmax > 0:
                x1 = track.x + track.w * min(1.0, 1.0 / vmax)
                pygame.draw.line(surf, theme.MUTED, (x1, track.y - 1), (x1, track.bottom), 1)
            f.text(surf, analysis.format_value(metric, v), 14, theme.TEXT,
                   (r.right - pad, track.centery), "midright", mono=True)
            y += bar_h

    def _draw_legend(self, surf, r: pygame.Rect, stats) -> None:
        f = self.fonts
        x = r.x
        for meta, col, _ in stats:
            pygame.draw.rect(surf, col, (x, r.centery - f.px(5), f.px(22), f.px(10)), border_radius=f.px(3))
            name = meta.get("name") or "untitled"
            created = datetime.fromtimestamp(meta.get("created", 0))
            x = f.text(surf, f"{name}  {created:%H:%M}  {_clock(meta.get('duration_s', 0))}", 14, theme.MUTED,
                       (x + f.px(28), r.centery), "midleft").right + f.px(22)
        f.text(surf, "spectra: dB, shared scale · shaded = alpha · lines = flicker frequencies", 13, theme.DIM,
               (r.right, r.centery), "midright")

    # -- modals -------------------------------------------------------------------------------
    def _shade(self, surf) -> None:
        shade = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 160))
        surf.blit(shade, (0, 0))

    def _text_field(self, surf, field: pygame.Rect, text: str, placeholder: str) -> None:
        f = self.fonts
        theme.panel(surf, field, theme.BG, theme.TEXT, f.px(8))
        cursor = "|" if int(time.monotonic() * 2) % 2 == 0 else " "
        f.text(surf, text + cursor, 20, theme.TEXT, (field.x + f.px(12), field.centery), "midleft",
               max_width=field.w - f.px(24))
        if not text:
            f.text(surf, placeholder, 17, theme.DIM, (field.x + f.px(24), field.centery), "midleft",
                   max_width=field.w - f.px(36))

    def _draw_prompt(self, surf) -> None:
        f = self.fonts
        w, h = surf.get_size()
        self._shade(surf)
        box = pygame.Rect(0, 0, f.px(680), f.px(190))
        box.center = (w // 2, h // 2)
        theme.panel(surf, box, theme.PANEL, theme.EDGE, f.px(12))
        f.text(surf, self.prompt["title"], 22, theme.TEXT, (box.x + f.px(24), box.y + f.px(22)), bold=True)
        field = pygame.Rect(box.x + f.px(24), box.y + f.px(66), box.w - f.px(48), f.px(46))
        self._text_field(surf, field, self.prompt["text"], "e.g. eyes-closed, flex-at-Oz, staring-12Hz")
        f.text(surf, "Enter save · Esc keep as is", 15, theme.MUTED, (box.x + f.px(24), box.bottom - f.px(22)),
               "midleft")

    def _draw_picker(self, surf) -> None:
        f = self.fonts
        pk = self.picker
        w, h = surf.get_size()
        self._shade(surf)
        box = pygame.Rect(0, 0, min(f.px(1000), w - f.px(20)), min(f.px(780), h - f.px(20)))
        box.center = (w // 2, h // 2)
        self._modal_box = box
        theme.panel(surf, box, theme.PANEL, theme.EDGE, f.px(12))
        pad = f.px(24)
        f.text(surf, f"Where is the {pk['label']} lead {pk['sdk']}?", 22, theme.TEXT, (box.x + pad, box.y + f.px(20)),
               bold=True)
        if pk["kind"] == "live":
            sub = f"Now {pk['current']}. Click its spot (from now on; remembered next time)."
        else:
            sub = f"Recorded as {pk['current']}. Click where it really was: this recording is corrected."
        f.text(surf, sub, 15, theme.MUTED, (box.x + pad, box.y + f.px(52)), max_width=box.w - 2 * pad)
        f.text(surf, f"Yellow ring = another {pk['label']} lead (the two swap). Dim = the other device's.", 15,
               theme.DIM, (box.x + pad, box.y + f.px(72)), max_width=box.w - 2 * pad)

        side_w = f.px(270)
        bottom_h = f.px(100)
        head_area = pygame.Rect(box.x + pad, box.y + f.px(100), box.w - 3 * pad - side_w,
                                box.h - f.px(100) - bottom_h)
        R = min(head_area.h / 2.45, head_area.w / 2.5)
        cx = head_area.centerx
        cy = head_area.y + 1.16 * R
        pygame.draw.circle(surf, theme.EDGE, (round(cx), round(cy)), round(R), 2)
        nose = [(cx - R * 0.12, cy - R * 0.99), (cx, cy - R * 1.12), (cx + R * 0.12, cy - R * 0.99)]
        pygame.draw.lines(surf, theme.EDGE, False, nose, 2)
        for side in (-1, 1):
            ear = pygame.Rect(0, 0, R * 0.12, R * 0.34)
            ear.center = (round(cx + side * R * 1.04), round(cy))
            pygame.draw.ellipse(surf, theme.EDGE, ear, 2)
        dot_r = max(f.px(13), min(f.px(19), round(R * 0.085)))
        tip = None
        for pos, (hx, hy) in placement.HEAD.items():
            c = (round(cx + hx * R), round(cy - hy * R))
            rect = pygame.Rect(0, 0, 2 * dot_r, 2 * dot_r)
            rect.center = c
            owner = pk["taken"].get(pos)
            current = pos == pk["current"]
            blocked = owner is not None and not owner[1]
            over = rect.collidepoint(self._mouse)
            hover = over and not current and not blocked and self._hover(rect)
            if current:
                fill, edge, ink = theme.TRAIL, None, theme.INK
            elif blocked:
                fill, edge, ink = (30, 34, 44), None, theme.DIM
            elif owner is not None:
                fill, edge, ink = (theme.WARN, None, theme.INK) if hover else (theme.PANEL_HI, theme.WARN, theme.TEXT)
            else:
                fill, edge, ink = (theme.TEXT, None, theme.INK) if hover else (theme.BG, theme.EDGE, theme.MUTED)
            pygame.draw.circle(surf, fill, c, dot_r)
            if edge is not None:
                pygame.draw.circle(surf, edge, c, dot_r, 2)
            f.text(surf, pos, 13, ink, c, "center", bold=True)
            if over:
                tip = (c, "this lead" if current else f"{owner[0]}: taken" if blocked else
                       f"swap with {owner[0]}" if owner else f"move here")
            if not current and not blocked:
                self._modal_hits.append((rect, lambda p=pos: self.apply_picker(p)))
        if tip is not None:
            f.text(surf, tip[1], 14, theme.TEXT, (tip[0][0], tip[0][1] + dot_r + f.px(3)), "midtop", bold=True)
        if pk["current"] not in placement.HEAD:
            f.text(surf, f"(now a custom name: {pk['current']})", 14, theme.TRAIL, (cx, cy + R * 1.3), "midtop")

        # every lead, right
        side = pygame.Rect(box.right - pad - side_w, head_area.y, side_w, head_area.h)
        theme.panel(surf, side, theme.BG, None, f.px(8))
        y = side.y + f.px(12)
        f.text(surf, "WHERE EVERY LEAD IS", 14, theme.MUTED, (side.x + f.px(14), y), bold=True)
        y += f.px(28)
        for label, mine, pairs in pk["leads"]:
            f.text(surf, label.upper(), 15, theme.TEXT if mine else theme.MUTED, (side.x + f.px(14), y), bold=True)
            y += f.px(24)
            for c, p in pairs:
                this = mine and c == pk["sdk"]
                col = theme.TRAIL if this else theme.TEXT if mine else theme.DIM
                f.text(surf, f"lead {c}", 15, col, (side.x + f.px(24), y), bold=this)
                f.text(surf, p, 15, col, (side.x + f.px(120), y), bold=True)
                if this:
                    f.text(surf, "← this one", 13, theme.TRAIL, (side.right - f.px(12), y + f.px(2)), "topright")
                y += f.px(22)
            y += f.px(12)

        # custom name, reset, cancel
        fy = box.bottom - bottom_h + f.px(10)
        f.text(surf, "Somewhere else? Type a name, Enter:", 15, theme.MUTED, (box.x + pad, fy))
        field = pygame.Rect(box.x + pad, fy + f.px(24), f.px(300), f.px(40))
        self._text_field(surf, field, pk["text"], "e.g. Iz-left, mastoid")
        x = field.right + f.px(14)
        default = pk.get("default")
        if default and default != pk["current"] and default not in {p for p, o in pk["taken"].items() if not o[1]}:
            txt = f.render(f"Back to config.py: {default}", 15, theme.TEXT, bold=True)
            b = pygame.Rect(x, field.y, txt.get_width() + f.px(24), field.h)
            hov = self._hover(b)
            theme.panel(surf, b, theme.EDGE if hov else theme.PANEL_HI, theme.EDGE, f.px(8))
            surf.blit(txt, txt.get_rect(center=b.center))
            self._modal_hits.append((b, lambda d=default: self.apply_picker(d)))
        txt = f.render("Cancel  Esc", 15, theme.MUTED, bold=True)
        b = pygame.Rect(0, field.y, txt.get_width() + f.px(24), field.h)
        b.right = box.right - pad
        self._hover(b)
        theme.panel(surf, b, theme.PANEL, theme.EDGE, f.px(8))
        surf.blit(txt, txt.get_rect(center=b.center))
        self._modal_hits.append((b, lambda: setattr(self, "picker", None)))
