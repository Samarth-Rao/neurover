"""Decoder view (I): how the source tells the commands apart, live.

SSVEP (EEG): one row per flicker box.
    response   per headset: how much that box's frequency (and its double) stands out from the
               neighbouring frequencies in the latest window. ~1x means nothing; staring at the box
               should push it to 2x or more. This is the raw evidence, before any decoder.
    score      the decoder's score for that box over the last 10 s, with the decision threshold.
               The box with the highest score above the line (and clear of the runner-up) wins.
EMG (Myo): one row per class: the classifier's probability over the last 10 s.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pygame

import commands

from . import theme
from .theme import Fonts

if TYPE_CHECKING:
    from .app import App

_ARROWS = {"up": "↑", "down": "↓", "left": "←", "right": "→"}
_RESP_MAX = 5.0          # the response bars run from 0.5x to 5x on a log scale


class DecodingView:
    def __init__(self, fonts: Fonts) -> None:
        self.fonts = fonts

    def draw(self, surf: pygame.Surface, area: pygame.Rect, app: "App") -> None:
        f = self.fonts
        src = app.source
        r = area.inflate(-f.px(24), -f.px(24))
        theme.panel(surf, r, fill=theme.PANEL, border=theme.EDGE, radius=f.px(12), width=2)
        pad = f.px(16)
        x, y = r.x + pad, r.y + pad
        eeg = getattr(src, "eeg", False) and hasattr(getattr(src, "decoder", None), "targets")
        title = "HOW THE BOXES ARE TOLD APART" if eeg else "HOW THE GESTURES ARE TOLD APART"
        tr = f.text(surf, title, 22, theme.TEXT, (x, y), bold=True, max_width=r.w - 2 * pad)
        f.text(surf, "I closes", 16, theme.DIM, (r.right - pad, tr.centery), "midright")
        y = tr.bottom + f.px(4)
        sub = ("Staring at a box should make its frequency stand out in the EEG (2x or more). The decoder "
               "picks the box whose score is above the line and clear of the rest; otherwise S."
               if eeg else "Each window is compared with the trained examples; the most similar gesture wins.")
        for line in theme.wrap(f, sub, 16, r.w - 2 * pad):
            f.text(surf, line, 16, theme.MUTED, (x, y))
            y += f.px(20)
        y += f.px(6)
        if eeg:
            self._ssvep(surf, pygame.Rect(x, y, r.w - 2 * pad, r.bottom - pad - y), app)
        else:
            self._history(surf, pygame.Rect(x, y, r.w - 2 * pad, r.bottom - pad - y), app, list(src.classes))

    # -- SSVEP ---------------------------------------------------------------------------------
    def _ssvep(self, surf, r: pygame.Rect, app: "App") -> None:
        f = self.fonts
        src = app.source
        targets = src.decoder.targets
        devices = [d for d in getattr(src, "device_names", ())]
        labels = getattr(src, "device_labels", {})
        resp = getattr(src, "response", {})
        n = len(targets)
        row_h = r.h / max(1, n)
        name_w = f.px(150)
        resp_w = int((r.w - name_w) * 0.45)
        hist_x = r.x + name_w + resp_w + f.px(16)
        hist_w = r.right - hist_x
        current = app.controller.raw.label if app.controller.raw is not None else None
        for i, (label, hz) in enumerate(targets.items()):
            top = r.y + row_h * i
            cy = top + row_h / 2
            cmd = src.command_map.get(label)
            if label == current:
                pygame.draw.rect(surf, theme.PANEL_HI, (r.x - f.px(6), round(top + 2), r.w + f.px(12),
                                                        round(row_h - 4)), border_radius=f.px(8))
            chip = pygame.Rect(r.x, 0, f.px(34), f.px(30))
            chip.centery = round(cy)
            pygame.draw.rect(surf, theme.cmd_color(cmd), chip, border_radius=f.px(5))
            f.text(surf, cmd or "?", 20, theme.INK, chip.center, "center", bold=True)
            f.text(surf, f"{_ARROWS.get(label, '')} {hz:.1f} Hz", 19, theme.TEXT, (chip.right + f.px(8), cy),
                   "midleft", bold=True)
            # response bars, one per headset
            bars = devices or list(resp)
            bh = max(f.px(10), min(f.px(18), (row_h - f.px(16)) / max(1, len(bars)) - f.px(4)))
            by = cy - (len(bars) * (bh + f.px(4))) / 2
            bx = r.x + name_w
            for dev in bars:
                v = resp.get(dev, {}).get(label)
                self._response_bar(surf, pygame.Rect(bx, round(by), resp_w, round(bh)), v, labels.get(dev, dev))
                by += bh + f.px(4)
            # score history
            self._sparkline(surf, pygame.Rect(hist_x, round(top + f.px(6)), hist_w, round(row_h - f.px(12))),
                            src, label, theme.cmd_color(cmd))
        # the scale legend under the response column
        f.text(surf, "response: 1x = nothing · 2x+ = a real response", 14, theme.DIM,
               (r.x + name_w, r.bottom), "bottomleft")
        f.text(surf, "score, last 10 s · line = threshold", 14, theme.DIM, (hist_x, r.bottom), "bottomleft")

    def _response_bar(self, surf, r: pygame.Rect, v, name: str) -> None:
        f = self.fonts
        label_w = f.px(78)
        f.text(surf, name, 14, theme.MUTED, (r.x, r.centery), "midleft", max_width=label_w - f.px(4))
        track = pygame.Rect(r.x + label_w, r.y, r.w - label_w - f.px(44), r.h)
        pygame.draw.rect(surf, theme.PANEL_HI, track, border_radius=f.px(3))

        def fx(ratio: float) -> int:
            lo, hi = math.log(0.5), math.log(_RESP_MAX)
            return track.x + round(track.w * (min(max(math.log(max(ratio, 1e-3)), lo), hi) - lo) / (hi - lo))

        if v is not None:
            col = theme.GOOD if v >= 2.0 else theme.WARN if v >= 1.4 else theme.DIM
            pygame.draw.rect(surf, col, (track.x, track.y, max(2, fx(v) - track.x), track.h), border_radius=f.px(3))
        for mark in (1.0, 2.0):
            mx = fx(mark)
            pygame.draw.line(surf, theme.TEXT if mark == 2.0 else theme.MUTED, (mx, track.top - 2), (mx, track.bottom + 1), 1)
        f.text(surf, "—" if v is None else f"{v:.1f}x", 15, theme.TEXT if v is not None else theme.DIM,
               (r.right, r.centery), "midright", bold=True)

    def _sparkline(self, surf, r: pygame.Rect, src, label: str, color) -> None:
        f = self.fonts
        pygame.draw.rect(surf, theme.PANEL_HI, r, border_radius=f.px(4))
        hist = list(getattr(src, "score_history", []))
        scale = float(getattr(src, "score_scale", 1.0) or 1.0)
        thr = float(getattr(src, "score_threshold", 0.0)) / scale

        def fy(v: float) -> float:
            return r.bottom - min(max(v, 0.0), 1.0) * (r.h - 2) - 1

        ty = fy(thr)
        pygame.draw.line(surf, theme.TEXT, (r.x, ty), (r.right, ty), 1)
        if len(hist) >= 2:
            t_end = hist[-1][0]
            pts = [(r.right - (t_end - t) / 10.0 * r.w, fy(s.get(label, 0.0) / scale)) for t, s, _ in hist
                   if t_end - t <= 10.0]
            if len(pts) >= 2:
                pygame.draw.lines(surf, color, False, pts, 2)
            now = hist[-1][1].get(label, 0.0)
            f.text(surf, f"{now:.2f}" if abs(now) < 10 else f"{now:.0f}", 14, theme.TEXT,
                   (r.right - f.px(4), r.y + f.px(2)), "topright", bold=True)

    # -- EMG: class probabilities over time -----------------------------------------------------------
    def _history(self, surf, r: pygame.Rect, app: "App", classes) -> None:
        f = self.fonts
        src = app.source
        names = getattr(src, "class_names", {})
        hist = list(getattr(app.controller, "raw_history", []))
        probs = (app.controller.raw.probs or {}) if app.controller.raw is not None else {}
        row_h = r.h / max(1, len(classes))
        for i, cls in enumerate(classes):
            top = r.y + row_h * i
            cy = top + row_h / 2
            cmd = commands.from_label(cls, src.command_map)
            chip = pygame.Rect(r.x, 0, f.px(34), f.px(30))
            chip.centery = round(cy)
            pygame.draw.rect(surf, theme.cmd_color(cmd), chip, border_radius=f.px(5))
            f.text(surf, cmd or "?", 20, theme.INK, chip.center, "center", bold=True)
            f.text(surf, names.get(cls, cls), 19, theme.TEXT, (chip.right + f.px(8), cy), "midleft", bold=True)
            track = pygame.Rect(r.x + f.px(170), round(cy - f.px(9)), r.w - f.px(240), f.px(18))
            pygame.draw.rect(surf, theme.PANEL_HI, track, border_radius=f.px(4))
            p = float(probs.get(cls, 0.0))
            if p > 0:
                pygame.draw.rect(surf, theme.cmd_color(cmd), (track.x, track.y, max(2, round(track.w * p)), track.h),
                                 border_radius=f.px(4))
            f.text(surf, f"{p * 100:.0f}%", 16, theme.TEXT, (r.right, cy), "midright", bold=True)
        if not hist:
            f.text(surf, "no predictions yet", 16, theme.DIM, r.center, "center")
