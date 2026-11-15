"""Right-hand dashboard: source status, D-pad, raw vs smoothed, confidence bars,
live signal bars, command timeline, sink status and FPS."""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import pygame

import commands
import config

from . import theme
from .theme import Fonts

if TYPE_CHECKING:
    from .app import App

# Design heights (px at 1600x900); scaled to fit the actual panel height.
_SECTIONS = (("header", 70), ("command", 200), ("prediction", 256), ("signal", 120),
             ("timeline", 100), ("footer", 70))
# EEG needs room for traces + spectrum: same sections, the signal one grows.
_SECTIONS_EEG = (("header", 70), ("command", 168), ("prediction", 214), ("signal", 250),
                 ("timeline", 88), ("footer", 62))
_GAP = 10
_ARROWS = {"up": "↑", "down": "↓", "left": "←", "right": "→"}


def _class_name(src, label: str) -> str:
    return src.class_names.get(label) or config.GESTURE_NAMES.get(label) or commands.NAMES.get(label, label).title()


def _score_threshold(app) -> float:
    """Decision threshold as a fraction of a full bar."""
    t = getattr(app.source, "score_threshold", None)
    return app.controller.smoother.threshold if t is None else t / _score_scale(app)


def _score_scale(app) -> float:
    """Score that fills a bar (EEG calibrated scores aren't 0-1)."""
    return float(getattr(app.source, "score_scale", 1.0) or 1.0)


class Dashboard:
    def __init__(self, fonts: Fonts) -> None:
        self.fonts = fonts

    def draw(self, surf: pygame.Surface, rect: pygame.Rect, app: "App") -> None:
        f = self.fonts
        pygame.draw.rect(surf, theme.BG, rect)
        pygame.draw.line(surf, theme.EDGE, rect.topleft, rect.bottomleft, 2)
        pad = f.px(16)
        inner = rect.inflate(-2 * pad, -2 * pad)
        sections = _SECTIONS_EEG if getattr(app.source, "eeg", False) else _SECTIONS
        gaps = f.px(_GAP) * (len(sections) - 1)
        total = sum(h for _, h in sections)
        k = (inner.h - gaps) / total
        y = inner.y
        for name, h in sections:
            r = pygame.Rect(inner.x, y, inner.w, round(h * k))
            getattr(self, f"_draw_{name}")(surf, r, app)
            y = r.bottom + f.px(_GAP)

    def draw_panels(self, surf: pygame.Surface, panels: dict, app: "App") -> None:
        """Dashboard split into the four corner panels of the SSVEP flicker layout."""
        f = self.fonts
        gap = f.px(_GAP)
        for rect in panels.values():
            pygame.draw.rect(surf, theme.BG, rect)
        tl = panels["tl"]
        head = pygame.Rect(tl.x, tl.y, tl.w, round(tl.h * 0.26))
        self._draw_header(surf, head, app)
        self._draw_command(surf, pygame.Rect(tl.x, head.bottom + gap, tl.w, tl.bottom - head.bottom - gap), app)
        self._draw_prediction(surf, panels["tr"], app)
        self._draw_signal(surf, panels["bl"], app)
        br = panels["br"]
        tline = pygame.Rect(br.x, br.y, br.w, round(br.h * 0.58))
        self._draw_timeline(surf, tline, app)
        self._draw_footer(surf, pygame.Rect(br.x, tline.bottom + gap, br.w, br.bottom - tline.bottom - gap), app)

    # -- sections ---------------------------------------------------------------------
    def _draw_header(self, surf, r: pygame.Rect, app: "App") -> None:
        f = self.fonts
        st = app.source.status()
        f.text(surf, "NEUROROVER", 30, theme.TEXT, r.topleft, bold=True)
        name = f"SOURCE: {app.source.name.upper()}"
        nr = f.text(surf, name, 28, theme.TEXT, r.topright, "topright", bold=True)
        col = theme.GOOD if st.ok else theme.WARN if st.pending else theme.BAD
        theme.dot(surf, col, (nr.left - f.px(16), nr.centery), f.px(9))
        f.text(surf, st.message, 21, theme.MUTED if st.ok else col, (r.right, nr.bottom + f.px(4)),
               "topright", bold=not st.ok, max_width=r.w)

    def _draw_command(self, surf, r: pygame.Rect, app: "App") -> None:
        f = self.fonts
        theme.panel(surf, r)
        ctl = app.controller
        sent = ctl.last_sent or commands.STOP
        ip = f.px(14)
        a = (r.h - 2 * ip) // 3
        cx, cy = r.x + ip + a * 1.5, r.centery
        self._dpad(surf, cx, cy, a, sent, ctl.blocked)

        x = r.x + ip * 2 + a * 3 + f.px(18)
        label = f.text(surf, "SENT TO ROVER", 18, theme.MUTED, (x, r.y + ip), bold=True)
        size = 150 * min(1.0, r.h / max(1, f.px(200)))
        letter = f.text(surf, sent, size, theme.cmd_color(sent), (x - f.px(4), (label.bottom + r.bottom) / 2),
                        "midleft", bold=True)
        tx = letter.right + f.px(16)
        f.text(surf, commands.NAMES[sent], 34, theme.TEXT, (tx, letter.centery - f.px(30)), "midleft", bold=True,
               max_width=r.right - tx - ip)
        f.text(surf, f"{config.SEND_HZ:g} Hz  ·  {ctl.sent_count:,} sent", 19, theme.MUTED,
               (tx, letter.centery + f.px(6)), "midleft", max_width=r.right - tx - ip)
        bx = tx
        by = letter.centery + f.px(36)
        if ctl.blocked:
            bx = self._pill(surf, "BLOCKED", theme.BAD, bx, by) + f.px(8)
        if ctl.paused:
            why = "CALIBRATING" if getattr(app, "calibration", None) is not None else "TRAINING"
            bx = self._pill(surf, f"PAUSED: {why}", theme.WARN, bx, by) + f.px(8)
        elif ctl.smoother.deadman:
            self._pill(surf, "DEAD-MAN", theme.BAD, bx, by)

    def _draw_prediction(self, surf, r: pygame.Rect, app: "App") -> None:
        f = self.fonts
        ctl, src = app.controller, app.source
        gap = f.px(10)
        card_h = round(r.h * 0.44)
        cw = (r.w - gap) // 2
        raw_r = pygame.Rect(r.x, r.y, cw, card_h)
        sm_r = pygame.Rect(r.x + cw + gap, r.y, r.w - cw - gap, card_h)

        # Raw prediction card
        ip = f.px(12)
        pred = ctl.raw
        if pred is None or pred.label is None:
            main, col = ("no model: press T" if src.supports_training and pred is not None else "—"), theme.DIM
        elif pred.label in commands.ALL:
            main, col = f"{pred.label}  {commands.NAMES[pred.label].title()}", theme.cmd_color(pred.label)
        else:
            main = f"{_class_name(src, pred.label)} → {ctl.raw_command or '?'}"
            col = theme.cmd_color(ctl.raw_command)
        conf = pred.confidence if pred is not None and pred.label is not None else 0.0
        thr = _score_threshold(app)
        low = pred is not None and conf / _score_scale(app) < thr
        aside = f"{src.decoder.score_name} {conf:.2f}" if getattr(src, "eeg", False) else f"{conf * 100:.0f}%"
        self._card(surf, raw_r, "RAW PREDICTION", aside, theme.WARN if low else theme.TEXT, main, col)
        # confidence bar with the threshold tick
        bar = pygame.Rect(raw_r.x + ip, raw_r.bottom - ip - f.px(10), raw_r.w - 2 * ip, f.px(10))
        pygame.draw.rect(surf, theme.PANEL_HI, bar, border_radius=f.px(3))
        frac = min(1.0, conf / _score_scale(app))
        if frac > 0:
            pygame.draw.rect(surf, col, (bar.x, bar.y, max(2, round(bar.w * frac)), bar.h), border_radius=f.px(3))
        tx = bar.x + round(bar.w * thr)
        pygame.draw.line(surf, theme.TEXT, (tx, bar.top - f.px(3)), (tx, bar.bottom + f.px(2)), 2)

        # Smoothed command card
        cmd = ctl.command
        sm = ctl.smoother
        n = sm.n_windows
        if sm.deadman:
            filled, dcol, note = 0, theme.BAD, "no data"
        elif sm.candidate is None:
            filled, dcol, note = 0, theme.DIM, "holding"
        elif sm.candidate == cmd:
            filled, dcol, note = n, theme.cmd_color(cmd), "locked"
        else:
            filled, dcol, note = min(sm.streak, n), theme.cmd_color(sm.candidate), f"{sm.candidate}?"
        self._card(surf, sm_r, "SMOOTHED", f"{min(filled, n)}/{n} {note}", theme.MUTED,
                   f"{cmd}  {commands.NAMES[cmd].title()}", theme.cmd_color(cmd))
        dr = f.px(7)
        dx = sm_r.x + ip + dr
        dy = sm_r.bottom - ip - f.px(5)
        for i in range(n):
            if i < filled:
                theme.dot(surf, dcol, (dx, dy), dr)
            else:
                pygame.draw.circle(surf, theme.DIM, (round(dx), round(dy)), dr, 2)
            dx += dr * 2 + f.px(6)

        # Per-class confidence bars
        bars = pygame.Rect(r.x, raw_r.bottom + gap, r.w, r.bottom - raw_r.bottom - gap)
        self._class_bars(surf, bars, app)

    def _class_bars(self, surf, r: pygame.Rect, app: "App") -> None:
        if getattr(getattr(app.source, "decoder", None), "mode", None) == "alpha":
            self._alpha_meter(surf, r, app)
            return
        f = self.fonts
        src, ctl = app.source, app.controller
        classes = list(src.classes)
        if not classes:
            return
        probs = (ctl.raw.probs or {}) if ctl.raw is not None else {}
        row_h = r.h / len(classes)
        name_w = f.px(116)
        chip_w = f.px(34)
        bar_x = r.x + name_w + chip_w + f.px(10)
        bar_w = r.right - bar_x - f.px(56)
        for i, cls in enumerate(classes):
            cy = r.y + row_h * (i + 0.5)
            cmd = commands.from_label(cls, src.command_map)
            label = _class_name(src, cls)
            f.text(surf, label, 20, theme.TEXT, (r.x, cy), "midleft", bold=True, max_width=name_w - f.px(4))
            chip = pygame.Rect(0, 0, chip_w - f.px(6), round(row_h * 0.72))
            chip.midleft = (r.x + name_w, cy)
            pygame.draw.rect(surf, theme.cmd_color(cmd), chip, border_radius=f.px(4))
            f.text(surf, cmd or "?", 18, theme.INK, chip.center, "center", bold=True)
            bh = max(4, round(row_h * 0.56))
            track = pygame.Rect(round(bar_x), round(cy - bh / 2), round(bar_w), bh)
            pygame.draw.rect(surf, theme.PANEL_HI, track, border_radius=f.px(3))
            p = float(probs.get(cls, 0.0))
            frac = min(1.0, p / _score_scale(app))
            if frac > 0:
                fill = track.copy()
                fill.w = max(2, round(track.w * frac))
                pygame.draw.rect(surf, theme.cmd_color(cmd), fill, border_radius=f.px(3))
            tx = track.x + round(track.w * _score_threshold(app))
            pygame.draw.line(surf, theme.TEXT, (tx, track.top - f.px(3)), (tx, track.bottom + f.px(2)), 2)
            value = (f"{p:.2f}" if abs(p) < 10 else f"{p:.0f}") if getattr(src, "eeg", False) else f"{p * 100:.0f}%"
            f.text(surf, value, 19, theme.TEXT if p else theme.DIM, (r.right, cy), "midright", bold=True)

    def _draw_signal(self, surf, r: pygame.Rect, app: "App") -> None:
        if getattr(app.source, "eeg", False):
            self._draw_eeg(surf, r, app)
            return
        f = self.fonts
        theme.panel(surf, r)
        ip = f.px(12)
        src = app.source
        if src.channels <= 0:
            f.text(surf, "SIGNAL", 16, theme.MUTED, (r.x + ip, r.y + ip), bold=True)
            f.text(surf, "No EMG for this source — keys map straight to commands", 20, theme.DIM,
                   (r.centerx, r.centery + f.px(8)), "center", max_width=r.w - 2 * ip)
            return
        f.text(surf, f"EMG  ·  {src.channels} CH", 16, theme.MUTED, (r.x + ip, r.y + ip), bold=True)
        pred = app.controller.raw
        emg = pred.emg if pred is not None else None
        full = float(getattr(src, "signal_full_scale", 1.0)) or 1.0
        # Per-channel scaling (relaxed level -> strong gesture) when the source provides it.
        levels = src.display_levels(emg) if emg is not None and hasattr(src, "display_levels") else None
        area = pygame.Rect(r.x + ip, r.y + f.px(34), r.w - 2 * ip, r.h - f.px(34) - ip - f.px(20))
        n = src.channels
        slot = area.w / n
        bw = slot * 0.66
        for i in range(n):
            x = area.x + slot * i + (slot - bw) / 2
            pygame.draw.rect(surf, theme.PANEL_HI, (round(x), area.y, round(bw), area.h), border_radius=f.px(3))
            if emg is not None and i < len(emg):
                v = float(levels[i]) if levels is not None else max(0.0, min(1.0, float(emg[i]) / full))
                h = round(area.h * v)
                if h > 0:
                    col = theme.lerp_color(theme.TRAIL, theme.WARN, v)
                    pygame.draw.rect(surf, col, (round(x), area.bottom - h, round(bw), h), border_radius=f.px(3))
            f.text(surf, str(i + 1), 17, theme.MUTED, (x + bw / 2, area.bottom + f.px(2)), "midtop", bold=True)

    def _draw_timeline(self, surf, r: pygame.Rect, app: "App") -> None:
        f = self.fonts
        theme.panel(surf, r)
        ip = f.px(12)
        span = config.TIMELINE_S
        f.text(surf, f"LAST {span:g} s", 16, theme.MUTED, (r.x + ip, r.y + ip), bold=True)
        now = app.now
        label_w = f.px(64)
        lane_x = r.x + ip + label_w
        lane_w = r.right - ip - lane_x
        lane_h = max(8, round((r.h - f.px(34) - ip - f.px(16)) / 2))
        ctl = app.controller
        lanes = (("RAW", [(t, c, False) for t, c in ctl.raw_history]), ("SENT", list(ctl.sent_history)))
        y = r.y + f.px(32)
        px_per_s = lane_w / span
        for name, hist in lanes:
            f.text(surf, name, 17, theme.MUTED, (r.x + ip, y + lane_h / 2), "midleft", bold=True)
            pygame.draw.rect(surf, theme.PANEL_HI, (lane_x, y, lane_w, lane_h))
            for i, (t, c, blocked) in enumerate(hist):
                t_end = hist[i + 1][0] if i + 1 < len(hist) else now
                x0 = lane_x + lane_w - (now - t) * px_per_s
                x1 = lane_x + lane_w - (now - t_end) * px_per_s
                x0, x1 = max(x0, lane_x), min(x1, lane_x + lane_w)
                if x1 <= x0:
                    continue
                seg = pygame.Rect(round(x0), y, max(1, round(x1) - round(x0)), lane_h)
                pygame.draw.rect(surf, theme.cmd_color(c), seg)
                if blocked:
                    pygame.draw.rect(surf, theme.BAD, (seg.x, seg.bottom - f.px(5), seg.w, f.px(5)))
            y += lane_h + f.px(6)
        for s in range(0, int(span) + 1, 2):
            x = lane_x + lane_w - s * px_per_s
            f.text(surf, "now" if s == 0 else f"-{s}s", 15, theme.DIM, (x, y - f.px(2)),
                   "topright" if s == 0 else "midtop")

    def _draw_footer(self, surf, r: pygame.Rect, app: "App") -> None:
        f = self.fonts
        x = r.x
        y = r.y + f.px(16)
        for sink in app.sinks:
            st = sink.status()
            theme.dot(surf, theme.GOOD if st.ok else theme.BAD, (x + f.px(9), y), f.px(9))
            tr = f.text(surf, st.label, 22, theme.TEXT, (x + f.px(24), y), "midleft", bold=True)
            x = tr.right + f.px(8)
            if st.detail:
                tr = f.text(surf, st.detail, 18, theme.MUTED, (x, y), "midleft")
                x = tr.right
            x += f.px(22)
        if not app.sinks:
            theme.dot(surf, theme.BAD, (x + f.px(9), y), f.px(9))
            f.text(surf, "NO SINKS", 22, theme.BAD, (x + f.px(24), y), "midleft", bold=True)
        f.text(surf, app.fps_text(), 20, theme.MUTED, (r.right, y), "midright", bold=True)
        f.text(surf, app.key_hints(), 17, theme.DIM, (r.x, r.bottom), "bottomleft", max_width=r.w)

    def _alpha_meter(self, surf, r: pygame.Rect, app: "App") -> None:
        f = self.fonts
        src = app.source
        thr = src.decoder.threshold
        value = float(src.scores.get("closed", 0.0))
        lo, hi = config.ALPHA_BAND
        tlo, thi = config.ALPHA_TOTAL_BAND
        closed = value >= thr
        col = theme.cmd_color(src.command_map["closed"] if closed else src.command_map["open"])
        vr = f.text(surf, f"{value:.2f}", 30, col, (r.right, r.y), "topright", bold=True)
        f.text(surf, f"ALPHA RATIO {lo:g}–{hi:g} / {tlo:g}–{thi:g} Hz  ·  threshold {thr:.2f}", 16, theme.MUTED,
               (r.x, r.y + f.px(6)), bold=True, max_width=vr.left - r.x - f.px(8))
        bar = pygame.Rect(r.x, vr.bottom + f.px(4), r.w, f.px(26))
        pygame.draw.rect(surf, theme.PANEL_HI, bar, border_radius=f.px(6))
        if value > 0:
            pygame.draw.rect(surf, col, (bar.x, bar.y, max(3, round(bar.w * min(1.0, value))), bar.h),
                             border_radius=f.px(6))
        tx = bar.x + round(bar.w * thr)
        pygame.draw.line(surf, theme.TEXT, (tx, bar.top - f.px(5)), (tx, bar.bottom + f.px(5)), f.px(3))
        y = bar.bottom + f.px(8)
        f.text(surf, "eyes open → S", 20, theme.TEXT if not closed else theme.DIM, (r.x, y), bold=not closed)
        f.text(surf, "eyes closed → F", 20, theme.TEXT if closed else theme.DIM, (r.right, y), "topright",
               bold=closed)

    # -- EEG --------------------------------------------------------------------------
    def _draw_eeg(self, surf, r: pygame.Rect, app: "App") -> None:
        f = self.fonts
        src = app.source
        theme.panel(surf, r)
        ip = f.px(12)
        lo, hi = config.EEG_DISPLAY_BAND
        f.text(surf, f"EEG  ·  {src.channels} CH  ·  {lo:g}–{hi:g} Hz", 16, theme.MUTED, (r.x + ip, r.y + ip),
               bold=True)
        top = r.y + f.px(36)
        body_h = r.bottom - ip - top
        traces_w = int((r.w - 3 * ip) * 0.58)
        f.text(surf, "OCCIPITAL SPECTRUM", 16, theme.MUTED, (r.x + 2 * ip + traces_w, r.y + ip), bold=True)
        self._eeg_traces(surf, pygame.Rect(r.x + ip, top, traces_w, body_h), src)
        self._eeg_spectrum(surf, pygame.Rect(r.x + 2 * ip + traces_w, top, r.w - 3 * ip - traces_w, body_h), src)
        if src.artifact:
            label = "ARTIFACT: window dropped"
            w = f.render(label, 19, theme.INK, bold=True).get_width() + f.px(18)
            self._pill(surf, label, theme.BAD, r.x + ip + traces_w - w, top + f.px(4))
        elif getattr(src, "blanked", 0.0) > 0:
            label = f"PATCHED {src.blanked * 100:.0f}%"   # spike / muscle burst left out, rest decoded
            w = f.render(label, 19, theme.INK, bold=True).get_width() + f.px(18)
            self._pill(surf, label, theme.WARN, r.x + ip + traces_w - w, top + f.px(4))

    def _eeg_traces(self, surf, r: pygame.Rect, src) -> None:
        f = self.fonts
        data = src.traces()
        n, c = data.shape
        if c == 0 or n < 2:
            return
        row_h = (r.h - f.px(16)) / c
        label_w = f.px(62)
        plot_x, plot_w = r.x + label_w, r.w - label_w
        step = max(1, n // max(2, plot_w))
        xs = plot_x + np.arange(0, n, step) * (plot_w / n)
        uv = config.EEG_TRACE_UV
        col = theme.BAD if src.artifact else theme.TRAIL
        for i in range(c):
            cy = r.y + row_h * (i + 0.5)
            pos = src.positions[i]
            contact = src.contact.get(pos)
            q = getattr(src, "quality", {}).get(pos, "unknown")
            used = i in getattr(src, "decode_idx", range(c))
            if q == "bad" or contact is False:
                dot_col = theme.BAD                      # bad electrodes are red whether used or not
            elif q == "marginal":
                dot_col = theme.WARN                     # a bit noisy, still used
            elif q == "good":
                dot_col = theme.GOOD if used else theme.MUTED
            else:
                dot_col = theme.DIM
            theme.dot(surf, dot_col, (r.x + f.px(8), cy), f.px(6))
            ohm = getattr(src, "resistance", {}).get(pos)
            note = getattr(src, "quality_note", {}).get(pos) if q in ("bad", "marginal") else None
            name_col = theme.TEXT if used else theme.MUTED
            if ohm is None and note is None:
                f.text(surf, pos, 20, name_col, (r.x + f.px(20), cy), "midleft", bold=True)
            else:  # the problem (or the measured resistance) under the position name
                f.text(surf, pos, 18, name_col, (r.x + f.px(20), cy + f.px(1)), "bottomleft", bold=True)
                if note is not None:
                    txt = note.split(" ")[0]
                else:
                    txt = "open" if ohm == float("inf") else f"{ohm / 1e6:.1f}MΩ" if ohm >= 1e5 else f"{ohm / 1e3:.0f}kΩ"
                f.text(surf, txt, 13, dot_col, (r.x + f.px(20), cy + f.px(1)), "topleft", bold=True)
            pygame.draw.line(surf, theme.PANEL_HI, (plot_x, cy), (plot_x + plot_w, cy), 1)
            ys = cy - np.clip(data[::step, i] / uv, -1.0, 1.0) * (row_h * 0.46)
            pygame.draw.lines(surf, col, False, np.column_stack([xs, ys]).tolist(), 1)
        f.text(surf, f"±{uv:g} µV  ·  {config.EEG_TRACE_S:g} s   ● contact", 15, theme.DIM, (r.right, r.bottom),
               "bottomright")

    def _eeg_spectrum(self, surf, r: pygame.Rect, src) -> None:
        f = self.fonts
        pygame.draw.rect(surf, theme.PANEL_HI, r, border_radius=f.px(4))
        lo, hi = config.EEG_SPECTRUM_HZ
        plot = pygame.Rect(r.x + f.px(4), r.y + f.px(20), r.w - f.px(8), r.h - f.px(40))

        def fx(freq: float) -> float:
            return plot.x + (freq - lo) / (hi - lo) * plot.w

        pygame.draw.rect(surf, (40, 46, 60), (fx(8), plot.y, fx(12) - fx(8), plot.h))  # alpha band
        for label, freq in getattr(src.decoder, "targets", {}).items():
            cmd = src.command_map.get(label)
            x = fx(freq)
            pygame.draw.line(surf, theme.cmd_color(cmd), (x, plot.y), (x, plot.bottom), 2)
            f.text(surf, cmd or "", 14, theme.cmd_color(cmd), (x, plot.y - f.px(1)), "midbottom", bold=True)
        for tick in (10, 20, 30, 40):
            if lo <= tick <= hi:
                f.text(surf, f"{tick}", 14, theme.DIM, (fx(tick), plot.bottom + f.px(2)), "midtop")
        f.text(surf, "Hz", 14, theme.DIM, (plot.x, plot.bottom + f.px(2)), "topleft")
        if src.spectrum is None:
            f.text(surf, "waiting for data…", 16, theme.DIM, plot.center, "center")
            return
        freqs, psd = src.spectrum
        sel = (freqs >= lo) & (freqs <= hi)
        if sel.sum() < 2:
            return
        db = 10 * np.log10(psd[sel] + 1e-6)
        top = float(db.max()) + 3.0
        span = 36.0
        ys = plot.bottom - np.clip((db - (top - span)) / span, 0.0, 1.0) * plot.h
        xs = [fx(v) for v in freqs[sel]]
        pygame.draw.lines(surf, theme.TEXT, False, list(zip(xs, ys.tolist())), 2)

    # -- widgets ----------------------------------------------------------------------
    def _card(self, surf, r: pygame.Rect, title: str, aside: str, aside_col, main: str, main_col) -> None:
        f = self.fonts
        ip = f.px(12)
        theme.panel(surf, r)
        tr = f.text(surf, title, 16, theme.MUTED, (r.x + ip, r.y + ip), bold=True)
        f.text(surf, aside, 18, aside_col, (r.right - ip, tr.centery), "midright", bold=True,
               max_width=r.right - ip - tr.right - f.px(10))
        f.text(surf, main, 30, main_col, (r.x + ip, (tr.bottom + r.bottom - f.px(24)) / 2), "midleft", bold=True,
               max_width=r.w - 2 * ip)

    def _dpad(self, surf, cx: float, cy: float, a: int, active: str, blocked: bool) -> None:
        f = self.fonts
        cells = {
            commands.FORWARD: (0, -1),
            commands.BACK: (0, 1),
            commands.LEFT: (-1, 0),
            commands.RIGHT: (1, 0),
            commands.STOP: (0, 0),
        }
        gap = max(2, f.px(4))
        for cmd, (gx, gy) in cells.items():
            rect = pygame.Rect(0, 0, a - gap, a - gap)
            rect.center = (round(cx + gx * a), round(cy + gy * a))
            on = cmd == active
            if on:
                fill = theme.BAD if (blocked and cmd == commands.FORWARD) else theme.cmd_color(cmd)
                pygame.draw.rect(surf, fill, rect, border_radius=f.px(10))
                fg = theme.INK
            else:
                pygame.draw.rect(surf, theme.PANEL_HI, rect, border_radius=f.px(10))
                pygame.draw.rect(surf, theme.EDGE, rect, width=2, border_radius=f.px(10))
                fg = theme.DIM
            f.text(surf, cmd, 40, fg, rect.center, "center", bold=True)

    def _pill(self, surf, text: str, color, x: float, cy: float) -> int:
        f = self.fonts
        t = f.render(text, 19, theme.INK, bold=True)
        r = pygame.Rect(round(x), 0, t.get_width() + f.px(18), t.get_height() + f.px(6))
        r.centery = round(cy)
        pygame.draw.rect(surf, color, r, border_radius=r.h // 2)
        surf.blit(t, t.get_rect(center=r.center))
        return r.right
