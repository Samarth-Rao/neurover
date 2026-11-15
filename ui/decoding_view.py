"""Decoder view (I): how the boxes are told apart, live. One row per flicker box:

    per headset   how far that box's frequency stands out from non-flicker frequencies on that headset
                  right now (z; ~0 = nothing, staring at the box should push it well above the line)
    pooled score  the headsets pooled, over the last 10 s, with the decision threshold. The box above
                  the line and clear of the runner-up wins; otherwise S.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pygame

from eeg.ssvep import ARROWS

from . import theme
from .theme import Fonts

if TYPE_CHECKING:
    from .app import App

_Z_MAX = 8.0          # bars run from -2 to this


class DecodingView:
    def __init__(self, fonts: Fonts) -> None:
        self.fonts = fonts

    def draw(self, surf: pygame.Surface, area: pygame.Rect, app: "App") -> None:
        f = self.fonts
        r = area.inflate(-f.px(24), -f.px(24))
        theme.panel(surf, r, fill=theme.PANEL, border=theme.EDGE, radius=f.px(10), width=1)
        pad = f.px(18)
        x, y = r.x + pad, r.y + pad
        hint = f.render("close", 15, theme.MUTED, face=theme.TEXT_FACE)
        hint_r = hint.get_rect(topright=(r.right - pad, y + f.px(4)))
        cap = theme.keycap(f, surf, "I", 15, (hint_r.left - f.px(6), hint_r.centery), "midright")
        surf.blit(hint, hint_r)
        tr = f.text(surf, "How the boxes are told apart", 22, theme.TEXT, (x, y), face=theme.DEMI,
                    max_width=cap.left - f.px(16) - x)
        y = tr.bottom + f.px(2)
        sub = ("Staring at a box makes its frequency stand out (z) on each headset that picks it up. Headsets "
               "are pooled; the box above the line and clear of the rest wins, otherwise S.")
        for line in theme.wrap(f, sub, 16, min(r.w - 2 * pad, f.px(820)), face=theme.TEXT_FACE):
            f.text(surf, line, 16, theme.MUTED, (x, y), face=theme.TEXT_FACE)
            y += f.px(21)
        y += f.px(8)
        self._rows(surf, pygame.Rect(x, y, r.w - 2 * pad, r.bottom - pad - y), app)

    def _rows(self, surf, r: pygame.Rect, app: "App") -> None:
        f = self.fonts
        src = app.source
        devices = src.feeds.devices
        legend_h = f.px(22)
        n = max(1, len(src.labels))
        row_h = min((r.h - legend_h) / n, f.px(150))       # tall rows give the sparklines room, within reason
        name_w = f.px(150)
        bars_w = int((r.w - name_w) * 0.45)
        hist_x = r.x + name_w + bars_w + f.px(16)
        hist_w = r.right - hist_x
        current = app.controller.raw.label if app.controller.raw is not None else None
        for i, label in enumerate(src.labels):
            top = r.y + row_h * i
            cy = top + row_h / 2
            cmd = src.command_map.get(label)
            if label == current:
                pygame.draw.rect(surf, theme.PANEL_HI, (r.x - f.px(8), round(top + 2), r.w + f.px(16),
                                                        round(row_h - 4)), border_radius=f.px(6))
            chip = pygame.Rect(r.x, 0, f.px(30), f.px(28))
            chip.centery = round(cy)
            pygame.draw.rect(surf, theme.cmd_color(cmd), chip, border_radius=f.px(4))
            f.text(surf, cmd or "?", 18, theme.INK, (chip.centerx, chip.centery + 1), "center", face=theme.DATA)
            f.text(surf, f"{ARROWS.get(label, '')} {src.targets[label]:.1f} Hz", 19, theme.TEXT,
                   (chip.right + f.px(10), cy + 1), "midleft", face=theme.DATA, max_width=name_w - chip.w - f.px(14))
            bh = max(f.px(10), min(f.px(16), (row_h - f.px(16)) / max(1, len(devices)) - f.px(6)))
            by = cy - (len(devices) * (bh + f.px(6)) - f.px(6)) / 2
            for d in devices:
                z = src.decoders[d.name].z
                v = None if z is None else float(z[i])
                self._bar(surf, pygame.Rect(r.x + name_w, round(by), bars_w, round(bh)), v, d.label,
                          src.threshold)
                by += bh + f.px(6)
            self._sparkline(surf, pygame.Rect(hist_x, round(top + f.px(6)), hist_w, round(row_h - f.px(12))),
                            src, label, theme.cmd_color(cmd))
        legend = r.y + row_h * n + f.px(6)
        f.text(surf, "Per headset: z now (— = not decoding)", 14, theme.DIM, (r.x + name_w, legend), face=theme.TEXT_FACE,
               max_width=hist_x - f.px(16) - (r.x + name_w))
        f.text(surf, "Pooled, last 10 s; the line is the threshold", 14, theme.DIM, (hist_x, legend),
               face=theme.TEXT_FACE, max_width=hist_w)

    def _bar(self, surf, r: pygame.Rect, v, name: str, threshold: float) -> None:
        f = self.fonts
        label_w = f.px(78)
        f.text(surf, name, 14, theme.MUTED, (r.x, r.centery), "midleft", face=theme.MEDIUM, max_width=label_w - f.px(4))
        track = pygame.Rect(r.x + label_w, r.y, r.w - label_w - f.px(44), r.h)
        pygame.draw.rect(surf, theme.PANEL_HI, track, border_radius=f.px(3))

        def fx(z: float) -> int:
            return track.x + round(track.w * (min(max(z, -2.0), _Z_MAX) + 2.0) / (_Z_MAX + 2.0))

        zero = fx(0.0)
        if v is not None:
            col = theme.GOOD if v >= threshold else theme.WARN if v >= threshold / 2 else theme.DIM
            a, b = sorted((zero, fx(v)))
            pygame.draw.rect(surf, col, (a, track.y, max(2, b - a), track.h), border_radius=f.px(3))
        pygame.draw.line(surf, theme.MUTED, (zero, track.top - 2), (zero, track.bottom + 1), 1)
        tx = fx(threshold)
        pygame.draw.line(surf, theme.TEXT, (tx, track.top - 2), (tx, track.bottom + 1), 1)
        f.text(surf, "—" if v is None else f"{v:+.1f}", 15, theme.TEXT if v is not None else theme.DIM,
               (r.right, r.centery + 1), "midright", face=theme.DATA)

    def _sparkline(self, surf, r: pygame.Rect, src, label: str, color) -> None:
        f = self.fonts
        pygame.draw.rect(surf, theme.PANEL_HI, r, border_radius=f.px(4))
        hist = list(src.score_history)
        scale = max(2.0 * src.threshold, 1e-6)

        def fy(v: float) -> float:
            return r.bottom - min(max(v / scale, 0.0), 1.0) * (r.h - 2) - 1

        ty = fy(src.threshold)
        pygame.draw.line(surf, theme.TEXT, (r.x, ty), (r.right, ty), 1)
        if len(hist) >= 2:
            t_end = hist[-1][0]
            pts = [(r.right - (t_end - t) / 10.0 * r.w, fy(s.get(label, 0.0))) for t, s, _ in hist if t_end - t <= 10.0]
            if len(pts) >= 2:
                pygame.draw.lines(surf, color, False, pts, 2)
            f.text(surf, f"{hist[-1][1].get(label, 0.0):+.1f}", 14, theme.TEXT, (r.right - f.px(5), r.y + f.px(2)),
                   "topright", face=theme.DATA)
