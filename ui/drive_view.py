"""The middle of the window when there's no simulated arena (driving the real rover): what the driver
needs at a glance, as big as the space allows.

    Sent to rover   the command going out (D-pad + letter + name), and why the rover is held, if it is
                    (stop gate, dead-man, blocked)
    Box scores      each box's score now against the decision threshold, and its last 10 s
    Rover link      each sink's status, in full

Between the flicker boxes (short and wide) these are three columns; without the flicker (tall) the
command and link sit on top and the box scores get the rest, with room for their history. The dashboard
leaves these out while this is shown (Dashboard ``drive``), so nothing is shown twice. Not drawn during
calibration or fitting: the calibration overlay then has the middle to itself, as before."""
from __future__ import annotations

from typing import TYPE_CHECKING

import pygame

import commands
import config

from . import theme
from .dashboard import Dashboard, _class_name, _scale, _threshold_frac
from .theme import DATA, DEMI, TEXT_FACE, Fonts

if TYPE_CHECKING:
    from .app import App

_HISTORY_S = 10.0


class DriveView:
    def __init__(self, fonts: Fonts, dashboard: Dashboard) -> None:
        self.fonts = fonts
        self.dash = dashboard

    def draw(self, surf: pygame.Surface, rect: pygame.Rect, app: "App") -> None:
        f = self.fonts
        gap = f.px(12)
        if rect.h < f.px(480):      # between the flicker boxes: three columns
            inner = pygame.Rect(rect.x + f.px(24), rect.y + f.px(8), rect.w - 2 * f.px(24), rect.h - 2 * f.px(8))
            cmd_w, link_w = round(inner.w * 0.34), round(inner.w * 0.2)
            cmd = pygame.Rect(inner.x, inner.y, cmd_w, inner.h)
            link = pygame.Rect(inner.right - link_w, inner.y, link_w, inner.h)
            scores = pygame.Rect(cmd.right + gap, inner.y, link.left - gap - cmd.right - gap, inner.h)
            self._command(surf, cmd, app, letter=False)
            history = False         # bars only between the boxes: glanceable, and less motion by the flicker
        else:                       # the whole left side: command + link on top, scores below
            inner = rect.inflate(-2 * f.px(24), -2 * f.px(24))
            top_h = round(min(inner.h * 0.36, f.px(300)))
            link_w = round(inner.w * 0.3)
            cmd = pygame.Rect(inner.x, inner.y, inner.w - link_w - gap, top_h)
            link = pygame.Rect(cmd.right + gap, inner.y, link_w, top_h)
            scores = pygame.Rect(inner.x, cmd.bottom + gap, inner.w, inner.bottom - cmd.bottom - gap)
            self._command(surf, cmd, app, letter=True)
            history = True
        self._scores(surf, scores, app, history)
        self._link(surf, link, app)

    # -- sent to rover ---------------------------------------------------------------------
    def _command(self, surf, r: pygame.Rect, app: "App", letter: bool) -> None:
        f, d = self.fonts, self.dash
        theme.face(surf, r, f)
        ctl = app.controller
        sent = ctl.last_sent or commands.STOP
        ip = f.px(16)
        label = d._label(surf, "Sent to rover", (r.x + ip, r.y + ip))
        body = pygame.Rect(r.x + ip, label.bottom + f.px(8), r.w - 2 * ip, r.bottom - ip - label.bottom - f.px(8))
        text_min = f.px(150)
        a = max(f.px(18), min(body.h // 3, f.px(76), (body.w - text_min - f.px(20)) // 3))
        d._dpad(surf, body.x + a * 1.5, body.centery, a, sent, ctl.blocked)
        x = body.x + 3 * a + f.px(20)
        if letter:                  # the big letter too, when there's room beside the D-pad
            size = min(170, body.h / f.scale * 0.95)
            if body.right - x - f.width(sent, size, face=DATA) - f.px(18) >= text_min:
                lr = f.text(surf, sent, size, theme.cmd_color(sent), (x - f.px(4), body.centery), "midleft",
                            face=DATA)
                x = lr.right + f.px(18)
        tw = body.right - x
        big = 36 if body.h >= f.px(200) else 28
        name_h = f.font(big, face=DEMI).get_height()
        meta_h = f.font(16, face=TEXT_FACE).get_height()
        tags = d.tags(app)
        tag_h = f.font(17, face=DEMI).get_height() + f.px(4) if tags else 0
        block = name_h + meta_h + (f.px(10) + tag_h if tags else 0)
        y = max(body.y, body.centery - block // 2)
        name = f.text(surf, commands.NAMES[sent].title(), big, theme.TEXT, (x, y), face=DEMI, max_width=tw)
        meta = f.text(surf, f"{ctl.sent_count:,} sent at {config.SEND_HZ:g} Hz", 16, theme.MUTED,
                      (x, name.bottom), face=TEXT_FACE, max_width=tw)
        if tags:
            d.draw_tags(surf, app, x, meta.bottom + f.px(10), body.right, r.bottom - f.px(6), size=17)

    # -- box scores ------------------------------------------------------------------------
    def _scores(self, surf, r: pygame.Rect, app: "App", history: bool) -> None:
        f, d = self.fonts, self.dash
        theme.face(surf, r, f)
        src, ctl = app.source, app.controller
        ip = f.px(16)
        head = d._label(surf, "Box scores" if src.eeg else "Keys", (r.x + ip, r.y + ip))
        unit = "z, headsets pooled" if src.eeg else "share of key presses"
        last = f", and the last {_HISTORY_S:g} s" if history and hasattr(src, "score_history") else ""
        f.text(surf, f"{unit}{last}; the line is the threshold", 13, theme.DIM, (r.right - ip, head.centery),
               "midright", face=TEXT_FACE, max_width=r.right - ip - head.right - f.px(16))
        classes = [c for c in src.classes if c != "rest"]
        if not classes:
            return
        rows = pygame.Rect(r.x + ip, head.bottom + f.px(8), r.w - 2 * ip, r.bottom - ip - head.bottom - f.px(8))
        row_h = rows.h / len(classes)
        scores = (ctl.raw.scores or {}) if ctl.raw is not None else {}
        current = ctl.raw.label if ctl.raw is not None else None
        hist = list(getattr(src, "score_history", ()))
        scale, thr = _scale(app), _threshold_frac(app)
        size = max(12, min(24, row_h / f.scale * 0.42))
        chip = min(round(row_h * 0.62), f.px(34))
        name_w = f.width("← 20.0 Hz", size, face=DATA) + f.px(14) if src.eeg else f.px(size * 4.6)
        value_w = f.width("+10.0", size, face=DATA) + f.px(8)
        spark_w = round(rows.w * 0.36) if history and hist and rows.w >= f.px(420) else 0
        bar_x = rows.x + chip + f.px(12) + name_w
        bar_w = rows.right - bar_x - value_w - (spark_w + f.px(14) if spark_w else 0)
        for i, cls in enumerate(classes):
            top = rows.y + row_h * i
            cy = top + row_h / 2
            cmd = commands.from_label(cls, src.command_map)
            col = theme.cmd_color(cmd)
            if cls == current:      # the box in the lead right now
                pygame.draw.rect(surf, col, (r.x + f.px(5), round(top + row_h * 0.2), f.px(3), round(row_h * 0.6)),
                                 border_radius=f.px(2))
            cr = pygame.Rect(rows.x, 0, chip, chip)
            cr.centery = round(cy)
            pygame.draw.rect(surf, col, cr, border_radius=f.px(5))
            f.text(surf, cmd or "?", min(22, chip / f.scale * 0.62), theme.INK, (cr.centerx, cr.centery + 1),
                   "center", face=DATA)
            f.text(surf, _class_name(src, cls), size, theme.TEXT, (cr.right + f.px(12), cy + 1), "midleft",
                   face=DATA, max_width=name_w - f.px(6))
            bh = max(f.px(6), min(round(row_h * 0.34), f.px(22)))
            track = pygame.Rect(round(bar_x), round(cy - bh / 2), max(10, round(bar_w)), bh)
            pygame.draw.rect(surf, theme.PANEL_HI, track, border_radius=f.px(4))
            p = float(scores.get(cls, 0.0))
            frac = min(1.0, max(0.0, p / scale))
            if frac > 0:
                pygame.draw.rect(surf, col, (track.x, track.y, max(2, round(track.w * frac)), track.h),
                                 border_radius=f.px(4))
            tx = track.x + round(track.w * thr)
            pygame.draw.line(surf, theme.TEXT, (tx, track.top - f.px(4)), (tx, track.bottom + f.px(3)), 2)
            value = f"{p:+.1f}" if src.eeg else f"{p * 100:.0f}%"
            f.text(surf, value, size, theme.TEXT if p else theme.DIM, (track.right + value_w, cy + 1), "midright",
                   face=DATA)
            if spark_w:
                sh = max(f.px(14), min(round(row_h - f.px(14)), f.px(120)))
                self._sparkline(surf, pygame.Rect(rows.right - spark_w, round(cy - sh / 2), spark_w, sh),
                                hist, cls, col, scale, app.source.threshold)

    def _sparkline(self, surf, r: pygame.Rect, hist: list, label: str, color, scale: float,
                   threshold: float) -> None:
        f = self.fonts
        pygame.draw.rect(surf, theme.PANEL_HI, r, border_radius=f.px(4))

        def fy(v: float) -> float:
            return r.bottom - 1 - min(max(v / scale, 0.0), 1.0) * (r.h - 2)

        ty = round(fy(threshold))
        pygame.draw.line(surf, theme.MUTED, (r.x, ty), (r.right - 1, ty), 1)
        if len(hist) >= 2:
            t_end = hist[-1][0]
            pts = [(r.right - 1 - (t_end - t) / _HISTORY_S * (r.w - 2), fy(s.get(label, 0.0)))
                   for t, s, _ in hist if t_end - t <= _HISTORY_S]
            if len(pts) >= 2:
                pygame.draw.lines(surf, color, False, pts, 2)

    # -- rover link ------------------------------------------------------------------------
    def _link(self, surf, r: pygame.Rect, app: "App") -> None:
        f, d = self.fonts, self.dash
        theme.face(surf, r, f)
        ip = f.px(16)
        head = d._label(surf, "Rover link", (r.x + ip, r.y + ip))
        y = head.bottom + f.px(10)
        w = r.w - 2 * ip
        if not app.sinks:
            theme.dot(surf, theme.BAD, (r.x + ip + f.px(6), y + f.px(12)), f.px(6))
            f.text(surf, "No sinks", 20, theme.BAD, (r.x + ip + f.px(18), y + f.px(12)), "midleft", face=DEMI,
                   max_width=w - f.px(18))
            return
        size = 22 if r.h >= f.px(200) else 19
        for sink in app.sinks:
            st = sink.status()
            line_h = f.font(size, face=DATA).get_height()
            if y + line_h > r.bottom - ip + f.px(4):
                break
            theme.dot(surf, theme.GOOD if st.ok else theme.BAD, (r.x + ip + f.px(6), y + line_h / 2 - 1), f.px(6))
            lab = f.text(surf, st.label, size, theme.TEXT, (r.x + ip + f.px(20), y), face=DATA,
                         max_width=w - f.px(20))
            y = lab.bottom + f.px(2)
            if st.detail:
                room = max(1, (r.bottom - ip - y) // f.px(20))
                for line in theme.wrap(f, st.detail, 15, w - f.px(20), max_lines=min(3, room), face=TEXT_FACE):
                    t = f.text(surf, line, 15, theme.MUTED if st.ok else theme.BAD, (r.x + ip + f.px(20), y),
                               face=TEXT_FACE, max_width=w - f.px(20))
                    y = t.bottom - f.px(2)
            y += f.px(12)
