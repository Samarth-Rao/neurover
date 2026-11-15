"""Right-hand dashboard: source status, D-pad, raw vs smoothed, per-box scores, EEG traces +
spectrum, command timeline, then a status bar (sinks, frame timing) and the keys.

With ``drive`` (no simulated arena: the middle of the window shows the drive view, see drive_view.py)
the command, the per-box scores and the sinks are in the middle instead, so they're left out here.

Each section is an instrument face (a filled panel, no outline) with a quiet sentence-case label;
numbers and command letters are DIN, words are Avenir Next (see theme.py)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pygame

import commands
import config
from eeg.devices import ohm_text

from . import theme
from .theme import DATA, DEMI, MEDIUM, TEXT_FACE, Fonts

if TYPE_CHECKING:
    from .app import App

# Design heights (px at 1600x900); scaled to fit what's left once the status bar has its height.
_SECTIONS = (("header", 62), ("command", 196), ("prediction", 256), ("signal", 120), ("timeline", 100))
_SECTIONS_EEG = (("header", 62), ("command", 164), ("prediction", 222), ("signal", 244), ("timeline", 88))
_SECTIONS_DRIVE = (("header", 62), ("cards", 130), ("signal", 100), ("timeline", 150))
_SECTIONS_DRIVE_EEG = (("header", 62), ("cards", 112), ("signal", 500), ("timeline", 96))
_GAP = 10
_LABEL = 15            # section labels
_STATUS_COL = {"ok": theme.GOOD, "unknown": theme.DIM, "no data": theme.DIM}


def _class_name(src, label: str) -> str:
    return src.class_names.get(label) or commands.NAMES.get(label, label).title()


def _scale(app) -> float:
    """Score that fills a bar: twice the threshold for EEG (z scores), 1 for the keyboard."""
    return max(2.0 * app.source.threshold, 1e-6) if app.source.eeg else 1.0


def _threshold_frac(app) -> float:
    return app.source.threshold / _scale(app) if app.source.eeg else app.controller.smoother.threshold


class Dashboard:
    def __init__(self, fonts: Fonts) -> None:
        self.fonts = fonts
        self._keys_key = None
        self._keys_surf: pygame.Surface | None = None

    def draw(self, surf: pygame.Surface, rect: pygame.Rect, app: "App", drive: bool = False) -> None:
        f = self.fonts
        pygame.draw.rect(surf, theme.BG, rect)
        pygame.draw.line(surf, theme.EDGE, rect.topleft, rect.bottomleft, 1)
        pad = f.px(16)
        inner = rect.inflate(-2 * pad, -2 * pad)
        gap = f.px(_GAP)
        foot_h = self._footer_height(inner.w, app)
        if drive:
            sections = _SECTIONS_DRIVE_EEG if app.source.eeg else _SECTIONS_DRIVE
        else:
            sections = _SECTIONS_EEG if app.source.eeg else _SECTIONS
        body_h = inner.h - foot_h - gap
        k = (body_h - gap * (len(sections) - 1)) / sum(h for _, h in sections)
        y = inner.y
        for name, h in sections:
            r = pygame.Rect(inner.x, y, inner.w, round(h * k))
            getattr(self, f"_draw_{name}")(surf, r, app)
            y = r.bottom + gap
        self._draw_footer(surf, pygame.Rect(inner.x, inner.bottom - foot_h, inner.w, foot_h), app, sinks=not drive)

    def draw_panels(self, surf: pygame.Surface, panels: dict, app: "App", drive: bool = False) -> None:
        """Dashboard split into the four corner panels of the SSVEP flicker layout. With ``drive`` (the
        middle shows the command, box scores and rover link): header + raw / smoothed, spectrum, the
        electrodes, timeline + status."""
        f = self.fonts
        gap = f.px(_GAP)
        for rect in panels.values():
            pygame.draw.rect(surf, theme.BG, rect)
        tl = panels["tl"]
        head = pygame.Rect(tl.x, tl.y, tl.w, min(round(tl.h * 0.26), f.px(64)))
        self._draw_header(surf, head, app)
        rest = pygame.Rect(tl.x, head.bottom + gap, tl.w, tl.bottom - head.bottom - gap)
        if drive:
            self._draw_cards(surf, rest, app)
            self._spectrum_face(surf, panels["tr"], app)
            self._traces_face(surf, panels["bl"], app)
        else:
            self._draw_command(surf, rest, app)
            self._draw_prediction(surf, panels["tr"], app)
            self._draw_signal(surf, panels["bl"], app)
        br = panels["br"]
        foot_h = self._footer_height(br.w, app)
        tline = pygame.Rect(br.x, br.y, br.w, br.h - foot_h - gap)
        self._draw_timeline(surf, tline, app)
        self._draw_footer(surf, pygame.Rect(br.x, br.bottom - foot_h, br.w, foot_h), app, sinks=not drive)

    def _label(self, surf, text: str, pos, anchor: str = "topleft", color=theme.MUTED, max_width=None) -> pygame.Rect:
        return self.fonts.text(surf, text, _LABEL, color, pos, anchor, face=MEDIUM, max_width=max_width)

    # -- sections ---------------------------------------------------------------------
    def _draw_header(self, surf, r: pygame.Rect, app: "App") -> None:
        f = self.fonts
        st = app.source.status()
        mark = f.text(surf, "NEUROVER", 30, theme.TEXT, r.topleft, face=DATA)
        col = theme.GOOD if st.ok else theme.WARN if st.pending else theme.BAD
        name = f.text(surf, app.source.name.upper(), 24, theme.TEXT, (r.right, mark.bottom), "bottomright",
                      face=DATA, max_width=max(f.px(40), r.w - mark.w - f.px(60)))
        theme.dot(surf, col, (name.left - f.px(13), name.centery - f.px(2)), f.px(6))
        room = name.left - f.px(28) - mark.right - f.px(16)
        if f.width("Source", _LABEL, face=MEDIUM) <= room:
            self._label(surf, "Source", (name.left - f.px(28), name.bottom - f.px(2)), "bottomright")
        f.text(surf, st.message, 16, theme.MUTED if st.ok else col, (r.right, mark.bottom), "topright",
               face=TEXT_FACE if st.ok else DEMI, max_width=r.w)

    def _draw_command(self, surf, r: pygame.Rect, app: "App") -> None:
        f = self.fonts
        theme.face(surf, r, f)
        ctl = app.controller
        sent = ctl.last_sent or commands.STOP
        ip = f.px(14)
        a = (r.h - 2 * ip) // 3
        self._dpad(surf, r.x + ip + a * 1.5, r.centery, a, sent, ctl.blocked)
        x = r.x + ip * 2 + a * 3 + f.px(18)
        label = self._label(surf, "Sent to rover", (x, r.y + ip))
        size = 150 * min(1.0, r.h / max(1, f.px(200)))
        letter = f.text(surf, sent, size, theme.cmd_color(sent), (x - f.px(4), (label.bottom + r.bottom) / 2),
                        "midleft", face=DATA)
        tx = letter.right + f.px(18)
        tw = r.right - tx - ip
        name = f.text(surf, commands.NAMES[sent].title(), 30, theme.TEXT, (tx, letter.centery + f.px(2)),
                      "bottomleft", face=DEMI, max_width=tw)
        meta = f.text(surf, f"{ctl.sent_count:,} sent at {config.SEND_HZ:g} Hz", 16, theme.MUTED,
                      (tx, name.bottom + f.px(2)), face=TEXT_FACE, max_width=tw)
        self.draw_tags(surf, app, tx, meta.bottom + f.px(8), r.right - ip, r.bottom - f.px(4))

    @staticmethod
    def tags(app: "App") -> list[tuple[str, tuple[int, int, int]]]:
        """Why the rover is held, if it is: (text, colour) tags."""
        ctl = app.controller
        tags = []
        if ctl.blocked:
            tags.append(("Blocked", theme.BAD))
        if ctl.paused:
            tags.append(("Paused: calibrating", theme.WARN))
        elif ctl.smoother.deadman:
            tags.append(("Dead-man", theme.BAD))
        stopped = getattr(app.source, "stopped", None)     # the EEG stop gate: driver moving / clenching
        if stopped:
            tags.append((f"Stop: {stopped.lower()}", theme.WARN))
        return tags

    def draw_tags(self, surf, app: "App", x: int, y: int, right: int, bottom: int, size: float = 15) -> int:
        """The tags, flowing from (x, y) and wrapping within ``right`` / ``bottom``. Returns the bottom."""
        f = self.fonts
        bx, by, end = x, y, y
        line = self.fonts.font(size, face=DEMI).get_height() + f.px(4)
        for text, col in self.tags(app):
            w = self._tag_width(text, size)
            if bx > x and bx + w > right:
                bx, by = x, by + line + f.px(4)
            if by + line > bottom:
                break
            bx = self._tag(surf, text, col, bx, by, max_width=right - bx, size=size) + f.px(8)
            end = by + line
        return end

    def _draw_prediction(self, surf, r: pygame.Rect, app: "App") -> None:
        f = self.fonts
        gap = f.px(10)
        card_h = round(r.h * 0.48)
        self._draw_cards(surf, pygame.Rect(r.x, r.y, r.w, card_h), app)
        self._class_bars(surf, pygame.Rect(r.x, r.y + card_h + gap, r.w, r.h - card_h - gap), app)

    def _draw_cards(self, surf, r: pygame.Rect, app: "App") -> None:
        """Raw prediction and smoothed, side by side."""
        f = self.fonts
        ctl, src = app.controller, app.source
        gap = f.px(10)
        cw = (r.w - gap) // 2
        raw_r = pygame.Rect(r.x, r.y, cw, r.h)
        sm_r = pygame.Rect(r.x + cw + gap, r.y, r.w - cw - gap, r.h)

        pred = ctl.raw
        if pred is None or pred.label is None:
            main, col = "—", theme.DIM
        elif pred.label in commands.ALL:
            main, col = f"{pred.label}  {commands.NAMES[pred.label].title()}", theme.cmd_color(pred.label)
        else:
            main = f"{_class_name(src, pred.label)} → {ctl.raw_command or '?'}"
            col = theme.cmd_color(ctl.raw_command)
        conf = pred.confidence if pred is not None and pred.label is not None else 0.0
        thr = _threshold_frac(app)
        low = pred is not None and conf / _scale(app) < thr
        aside = f"z {conf:.1f}" if src.eeg else f"{conf * 100:.0f}%"
        ind = self._card(surf, raw_r, "Raw prediction", aside, theme.WARN if low else theme.TEXT, main, col)
        bar = pygame.Rect(ind.x, ind.centery - f.px(4), ind.w, f.px(8))
        pygame.draw.rect(surf, theme.PANEL_HI, bar, border_radius=f.px(3))
        frac = min(1.0, max(0.0, conf / _scale(app)))
        if frac > 0:
            pygame.draw.rect(surf, col, (bar.x, bar.y, max(2, round(bar.w * frac)), bar.h), border_radius=f.px(3))
        tx = bar.x + round(bar.w * thr)
        pygame.draw.line(surf, theme.TEXT, (tx, bar.top - f.px(3)), (tx, bar.bottom + f.px(2)), 2)

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
        ind = self._card(surf, sm_r, "Smoothed", f"{min(filled, n)}/{n} {note}", theme.MUTED,
                         f"{cmd}  {commands.NAMES[cmd].title()}", theme.cmd_color(cmd))
        dr = max(3, min(f.px(6), ind.h // 2, round((ind.w - f.px(4) * n) / max(1, 2 * n))))
        step = dr * 2 + max(2, min(f.px(6), round(ind.w / max(1, n)) - 2 * dr))
        dx, dy = ind.x + dr, ind.centery
        for i in range(n):
            if i < filled:
                theme.dot(surf, dcol, (dx, dy), dr)
            else:
                pygame.draw.circle(surf, theme.DIM, (round(dx), round(dy)), dr, 2)
            dx += step

    def _class_bars(self, surf, r: pygame.Rect, app: "App") -> None:
        f = self.fonts
        src, ctl = app.source, app.controller
        classes = [c for c in src.classes if c != "rest"]
        if not classes:
            return
        scores = (ctl.raw.scores or {}) if ctl.raw is not None else {}
        row_h = r.h / len(classes)
        name_w, chip_w = f.px(108), f.px(32)
        bar_x = r.x + name_w + chip_w + f.px(8)
        bar_w = r.right - bar_x - f.px(54)
        size = min(20, row_h / f.scale * 0.62)
        for i, cls in enumerate(classes):
            cy = r.y + row_h * (i + 0.5)
            cmd = commands.from_label(cls, src.command_map)
            f.text(surf, _class_name(src, cls), size, theme.TEXT, (r.x, cy), "midleft", face=DATA,
                   max_width=name_w - f.px(4))
            ch = min(round(row_h * 0.72), f.px(26))
            chip = pygame.Rect(0, 0, min(chip_w - f.px(6), ch + f.px(2)), ch)
            chip.midleft = (r.x + name_w, cy)
            pygame.draw.rect(surf, theme.cmd_color(cmd), chip, border_radius=f.px(4))
            f.text(surf, cmd or "?", min(17, size), theme.INK, (chip.centerx, chip.centery + 1), "center", face=DATA)
            bh = max(4, min(round(row_h * 0.5), f.px(16)))
            track = pygame.Rect(round(bar_x), round(cy - bh / 2), round(bar_w), bh)
            pygame.draw.rect(surf, theme.PANEL_HI, track, border_radius=f.px(3))
            p = float(scores.get(cls, 0.0))
            frac = min(1.0, max(0.0, p / _scale(app)))
            if frac > 0:
                fill = track.copy()
                fill.w = max(2, round(track.w * frac))
                pygame.draw.rect(surf, theme.cmd_color(cmd), fill, border_radius=f.px(3))
            tx = track.x + round(track.w * _threshold_frac(app))
            pygame.draw.line(surf, theme.TEXT, (tx, track.top - f.px(3)), (tx, track.bottom + f.px(2)), 2)
            value = f"{p:+.1f}" if src.eeg else f"{p * 100:.0f}%"
            f.text(surf, value, min(18, size), theme.TEXT if p else theme.DIM, (r.right, cy), "midright", face=DATA)

    def _draw_signal(self, surf, r: pygame.Rect, app: "App") -> None:
        f = self.fonts
        theme.face(surf, r, f)
        ip = f.px(12)
        src = app.source
        if not src.eeg:
            self._label(surf, "Signal", (r.x + ip, r.y + ip))
            f.text(surf, "Keyboard: keys map straight to commands", 18, theme.DIM,
                   (r.centerx, r.centery + f.px(8)), "center", face=TEXT_FACE, max_width=r.w - 2 * ip)
            return
        rows = src.electrodes()
        if r.h >= f.px(380):         # tall: electrodes over the spectrum, each the full width
            traces_h = round((r.h - 3 * ip) * 0.56)
            self._traces_body(surf, pygame.Rect(r.x + ip, r.y + ip, r.w - 2 * ip, traces_h), app, rows)
            self._spectrum_body(surf, pygame.Rect(r.x + ip, r.y + 2 * ip + traces_h, r.w - 2 * ip,
                                                  r.h - 3 * ip - traces_h), src)
            return
        traces_w = int((r.w - 3 * ip) * 0.58)
        self._traces_body(surf, pygame.Rect(r.x + ip, r.y + ip, traces_w, r.h - 2 * ip), app, rows)
        self._spectrum_body(surf, pygame.Rect(r.x + 2 * ip + traces_w, r.y + ip, r.w - 3 * ip - traces_w,
                                              r.h - 2 * ip), src)

    def _traces_face(self, surf, r: pygame.Rect, app: "App") -> None:
        theme.face(surf, r, self.fonts)
        ip = self.fonts.px(12)
        self._traces_body(surf, r.inflate(-2 * ip, -2 * ip), app, app.source.electrodes())

    def _spectrum_face(self, surf, r: pygame.Rect, app: "App") -> None:
        theme.face(surf, r, self.fonts)
        ip = self.fonts.px(12)
        self._spectrum_body(surf, r.inflate(-2 * ip, -2 * ip), app.source)

    def _traces_body(self, surf, r: pygame.Rect, app: "App", rows: list[dict]) -> None:
        """Label row ("EEG  4 electrodes, 1–40 Hz"), then one trace per electrode."""
        f = self.fonts
        lo, hi = config.EEG_DISPLAY_BAND
        head = self._label(surf, "EEG", r.topleft, color=theme.TEXT)
        n = len(rows)
        self._label(surf, f"{n} electrode{'s' if n != 1 else ''}, {lo:g}–{hi:g} Hz", (head.right + f.px(8), head.y),
                    max_width=r.right - head.right - f.px(8))
        top = head.bottom + f.px(6)
        self._eeg_traces(surf, pygame.Rect(r.x, top, r.w, r.bottom - top), app, rows)

    def _spectrum_body(self, surf, r: pygame.Rect, src) -> None:
        head = self._label(surf, "Spectrum (decoded)", r.topleft, max_width=r.w)
        top = head.bottom + self.fonts.px(6)
        self._eeg_spectrum(surf, pygame.Rect(r.x, top, r.w, r.bottom - top), src)

    def _eeg_traces(self, surf, r: pygame.Rect, app: "App", rows: list[dict]) -> None:
        f = self.fonts
        src = app.source
        c = len(rows)
        if c == 0:
            return
        note_h = f.font(13, face=TEXT_FACE).get_height()
        row_h = (r.h - note_h - f.px(2)) / c
        label_w = f.px(96)
        plot_x, plot_w = r.x + label_w, r.w - label_w
        span, uv = config.EEG_TRACE_S, config.EEG_TRACE_UV
        size = min(17, max(11, row_h / f.scale * 0.6))
        for i, row in enumerate(rows):
            cy = r.y + row_h * (i + 0.5)
            st = row["status"]
            dot = _STATUS_COL.get(st, theme.BAD) if row["used"] or st != "ok" else theme.MUTED
            theme.dot(surf, dot, (r.x + f.px(5), cy), max(2, min(f.px(5), round(row_h * 0.2))))
            f.text(surf, row["position"], size, theme.TEXT if row["used"] else theme.MUTED,
                   (r.x + f.px(15), cy), "midleft", face=DATA)
            if st not in ("ok", "unknown"):
                note, col = st, (theme.DIM if st == "no data" else theme.BAD)
            elif row["resistance"] is not None:
                good = row["resistance"] < config.EEG_CONTACT_GOOD_OHM
                note, col = ohm_text(row["resistance"]), theme.GOOD if good else theme.WARN
            else:
                note, col = ("" if row["level"] is None else f"{row['level']:.0f} µV"), theme.DIM
            f.text(surf, note, max(10, size - 4), col, (r.x + f.px(52), cy), "midleft", face=DATA,
                   max_width=label_w - f.px(56))
            pygame.draw.line(surf, theme.PANEL_HI, (plot_x, cy), (plot_x + plot_w, cy), 1)
            d = src.feeds.device(row["node"].device)
            t, x = d.window(None, span, "display")
            if len(t) < 2:
                continue
            t0 = app.now - span
            sel = t > t0
            if sel.sum() < 2:
                continue
            xs = plot_x + (t[sel] - t0) / span * plot_w
            ys = cy - np.clip(x[sel, row["col"]] / uv, -1.0, 1.0) * (row_h * 0.46)
            step = max(1, len(xs) // max(2, plot_w))
            pts = np.column_stack([xs[::step], ys[::step]])
            pygame.draw.lines(surf, theme.TRAIL if st in ("ok", "unknown") else theme.BAD, False, pts.tolist(), 1)
        scale = f.text(surf, f"±{uv:g} µV, {span:g} s", 13, theme.DIM, (r.x, r.bottom), "bottomleft", face=DATA)
        f.text(surf, "µV = level without the device's shared part", 13, theme.DIM, (r.right, r.bottom), "bottomright",
               face=TEXT_FACE, max_width=r.right - scale.right - f.px(12))

    def _eeg_spectrum(self, surf, r: pygame.Rect, src) -> None:
        f = self.fonts
        pygame.draw.rect(surf, theme.PANEL_HI, r, border_radius=f.px(4))
        lo, hi = config.EEG_SPECTRUM_HZ
        lab_h = f.font(14, face=DATA).get_height()

        def fx_in(plot_x: float, plot_w: float, freq: float) -> float:
            return plot_x + (freq - lo) / (hi - lo) * plot_w

        px0, pw = r.x + f.px(6), r.w - f.px(12)
        # Marker letters: a second row where two would touch (e.g. 7.5 and 8.57 Hz).
        marks, ends = [], []
        for label, freq in sorted(src.targets.items(), key=lambda kv: kv[1]):
            cmd = src.command_map.get(label)
            x = fx_in(px0, pw, freq)
            w = f.width(cmd or "", 14, face=DATA)
            row = 0
            while row < len(ends) and ends[row] > x - w / 2 - f.px(3):
                row += 1
            if row == len(ends):
                ends.append(0.0)
            ends[row] = x + w / 2
            marks.append((x, cmd, row))
        rows = max(1, len(ends))
        plot = pygame.Rect(px0, r.y + f.px(4) + rows * lab_h, pw, r.h - f.px(8) - rows * lab_h - lab_h)

        def fx(freq: float) -> float:
            return fx_in(plot.x, plot.w, freq)

        for x, cmd, row in marks:
            col = theme.cmd_color(cmd)
            pygame.draw.line(surf, col, (x, plot.y - (rows - 1 - row) * lab_h), (x, plot.bottom), 2)
            f.text(surf, cmd or "", 14, col, (x, plot.y - (rows - 1 - row) * lab_h), "midbottom", face=DATA)
        for tick in (10, 20, 30, 40):
            if lo <= tick <= hi:
                f.text(surf, f"{tick}", 13, theme.DIM, (fx(tick), plot.bottom + f.px(1)), "midtop", face=DATA)
        f.text(surf, "Hz", 13, theme.DIM, (plot.x, plot.bottom + f.px(1)), "topleft", face=DATA)
        if src.spectrum is None:
            f.text(surf, "Waiting for data…", 15, theme.DIM, plot.center, "center", face=TEXT_FACE,
                   max_width=plot.w)
            return
        freqs, psd = src.spectrum
        sel = (freqs >= lo) & (freqs <= hi)
        if sel.sum() < 2:
            return
        db = 10 * np.log10(psd[sel] + 1e-6)
        top, span = float(db.max()) + 3.0, 36.0
        ys = plot.bottom - np.clip((db - (top - span)) / span, 0.0, 1.0) * plot.h
        pygame.draw.lines(surf, theme.TEXT, False, list(zip([fx(v) for v in freqs[sel]], ys.tolist())), 2)

    def _draw_timeline(self, surf, r: pygame.Rect, app: "App") -> None:
        f = self.fonts
        theme.face(surf, r, f)
        ip = f.px(12)
        span = config.TIMELINE_S
        tick_h = f.font(13, face=DATA).get_height()
        title_h = f.font(_LABEL, face=MEDIUM).get_height()
        label_w = f.px(50)
        lane_x = r.x + ip + label_w
        lane_w = r.right - ip - lane_x
        lanes_bottom = r.bottom - ip - tick_h - f.px(2)
        top = r.y + ip
        if lanes_bottom - (top + title_h + f.px(6)) >= 2 * f.px(14) + f.px(5):     # room for a title row
            self._label(surf, f"Last {span:g} s", (r.x + ip, top))
            top += title_h + f.px(6)
        gap = f.px(5)
        lane_h = max(4, (lanes_bottom - top - gap) // 2)
        now = app.now
        ctl = app.controller
        lanes = (("Raw", [(t, c, False) for t, c in ctl.raw_history]), ("Sent", list(ctl.sent_history)))
        y = top
        px_per_s = lane_w / span
        for name, hist in lanes:
            self._label(surf, name, (r.x + ip, y + lane_h / 2), "midleft")
            pygame.draw.rect(surf, theme.PANEL_HI, (lane_x, y, lane_w, lane_h))
            for i, (t, c, blocked) in enumerate(hist):
                t_end = hist[i + 1][0] if i + 1 < len(hist) else now
                x0 = max(lane_x + lane_w - (now - t) * px_per_s, lane_x)
                x1 = min(lane_x + lane_w - (now - t_end) * px_per_s, lane_x + lane_w)
                if x1 <= x0:
                    continue
                seg = pygame.Rect(round(x0), y, max(1, round(x1) - round(x0)), lane_h)
                pygame.draw.rect(surf, theme.cmd_color(c), seg)
                if blocked:
                    pygame.draw.rect(surf, theme.BAD, (seg.x, seg.bottom - f.px(5), seg.w, f.px(5)))
            y += lane_h + gap
        ty = y - gap + f.px(2)
        for s in range(0, int(span) + 1, 2):
            x = lane_x + lane_w - s * px_per_s
            f.text(surf, "now" if s == 0 else f"−{s} s", 13, theme.DIM, (x, ty),
                   "topright" if s == 0 else "midtop", face=DATA)

    # -- status bar + keys --------------------------------------------------------------
    def _status_h(self) -> int:
        return self.fonts.font(17, face=DATA).get_height() + self.fonts.px(4)

    def _footer_height(self, w: int, app: "App") -> int:
        keys = self._keys(w, app)
        return self._status_h() + self.fonts.px(8) + keys.get_height()

    def _draw_footer(self, surf, r: pygame.Rect, app: "App", sinks: bool = True) -> None:
        f = self.fonts
        cy = r.y + self._status_h() // 2
        # Frame timing on the right, its width reserved; sinks fill what's left, shortening their details.
        parts = app.frame_status()
        x = r.right
        for text, level in reversed(parts):
            col = {"ok": theme.MUTED, "warn": theme.WARN, "bad": theme.BAD}[level]
            x = f.text(surf, text, 16, col, (x, cy), "midright", face=DATA).left - f.px(14)
        limit = x - f.px(10)
        chips = []
        for sink in app.sinks if sinks else ():
            st = sink.status()
            chips.append((st, f.px(18) + f.width(st.label, 17, face=DATA)))
        if not chips and sinks:
            theme.dot(surf, theme.BAD, (r.x + f.px(6), cy), f.px(6))
            f.text(surf, "No sinks", 17, theme.BAD, (r.x + f.px(18), cy), "midleft", face=DEMI,
                   max_width=limit - r.x - f.px(18))
        x = r.x
        chip_gap = f.px(20)
        for i, (st, fixed) in enumerate(chips):
            rest = sum(w for _, w in chips[i + 1:]) + chip_gap * (len(chips) - 1 - i)
            room = limit - x - rest
            if room < f.px(30):
                break
            theme.dot(surf, theme.GOOD if st.ok else theme.BAD, (x + f.px(6), cy), f.px(6))
            lab = f.text(surf, st.label, 17, theme.TEXT, (x + f.px(18), cy + 1), "midleft", face=DATA,
                         max_width=room - f.px(18))
            x = lab.right
            detail_room = room - fixed - f.px(8)
            if st.detail and detail_room >= f.px(36):
                x = f.text(surf, st.detail, 15, theme.MUTED if st.ok else theme.BAD, (x + f.px(8), cy), "midleft",
                           face=TEXT_FACE, max_width=detail_room).right
            x += chip_gap
        keys = self._keys(r.w, app)
        surf.blit(keys, (r.x, r.bottom - keys.get_height()))

    def _keys(self, w: int, app: "App") -> pygame.Surface:
        """The key hints as key caps, wrapped to ``w`` (never cut off). Built once per layout, then blitted."""
        f = self.fonts
        groups = app.key_hints()
        key = (repr(groups), w, f.scale)
        if key == self._keys_key and self._keys_surf is not None:
            return self._keys_surf
        size = 14
        cap_h = f.font(size, face=DATA).get_height() + f.px(2)
        row_h = cap_h + f.px(6)
        items = []                    # (width, keys, action, starts a group)
        for g, group in enumerate(groups):
            for i, (keys, action) in enumerate(group):
                caps = keys.split(" / ")
                iw = sum(f.width(k, size, face=DATA) + f.px(10) for k in caps) + f.px(14) * (len(caps) - 1)
                if action:
                    iw += f.px(5) + f.width(action, size, face=TEXT_FACE)
                items.append((iw, caps, action, g > 0 and i == 0))
        placed, x, row = [], 0, 0
        for iw, caps, action, new_group in items:
            gap = (f.px(24) if new_group else f.px(12)) if x else 0
            if x and x + gap + iw > w:
                x, row, gap = 0, row + 1, 0
            placed.append((x + gap, row, caps, action))
            x += gap + iw
        surf = pygame.Surface((max(1, w), row_h * (row + 1) - f.px(6)), pygame.SRCALPHA)
        for x, row, caps, action in placed:
            cy = row * row_h + cap_h / 2
            for j, k in enumerate(caps):
                if j:
                    x = f.text(surf, "/", size, theme.DIM, (x + f.px(4), cy), "midleft", face=TEXT_FACE).right + f.px(4)
                x = theme.keycap(f, surf, k, size, (x, cy)).right
            if action:
                f.text(surf, action, size, theme.MUTED, (x + f.px(5), cy), "midleft", face=TEXT_FACE)
        self._keys_key, self._keys_surf = key, surf
        return surf

    # -- widgets ----------------------------------------------------------------------
    def _card(self, surf, r: pygame.Rect, title: str, aside: str, aside_col, main: str, main_col) -> pygame.Rect:
        """A face with a label, a reading on the right, a big value, and a strip at the bottom for an
        indicator (returned)."""
        f = self.fonts
        ip = f.px(12)
        theme.face(surf, r, f)
        tr = self._label(surf, title, (r.x + ip, r.y + ip), max_width=r.w - 2 * ip - f.px(60))
        f.text(surf, aside, 16, aside_col, (r.right - ip, tr.centery + 1), "midright", face=DATA,
               max_width=r.right - ip - tr.right - f.px(10))
        ind = pygame.Rect(r.x + ip, r.bottom - ip - f.px(12), r.w - 2 * ip, f.px(12))
        space = ind.top - tr.bottom - f.px(4)
        size = max(12, min(30, space / f.scale / 1.1))       # DIN's caps are ~0.7 em: keep air above the strip
        f.text(surf, main, size, main_col, (r.x + ip, (tr.bottom + ind.top) / 2), "midleft", face=DATA,
               max_width=r.w - 2 * ip)
        return ind

    def _dpad(self, surf, cx: float, cy: float, a: int, active: str, blocked: bool) -> None:
        f = self.fonts
        cells = {commands.FORWARD: (0, -1), commands.BACK: (0, 1), commands.LEFT: (-1, 0),
                 commands.RIGHT: (1, 0), commands.STOP: (0, 0)}
        gap = max(2, f.px(4))
        size = min(36, a / f.scale * 0.6)
        for cmd, (gx, gy) in cells.items():
            rect = pygame.Rect(0, 0, a - gap, a - gap)
            rect.center = (round(cx + gx * a), round(cy + gy * a))
            if cmd == active:
                fill = theme.BAD if (blocked and cmd == commands.FORWARD) else theme.cmd_color(cmd)
                pygame.draw.rect(surf, fill, rect, border_radius=f.px(8))
                fg = theme.INK
            else:
                pygame.draw.rect(surf, theme.PANEL_HI, rect, border_radius=f.px(8))
                fg = theme.DIM
            f.text(surf, cmd, size, fg, (rect.centerx, rect.centery + 1), "center", face=DATA)

    def _tag_width(self, text: str, size: float = 15) -> int:
        return self.fonts.width(text, size, face=DEMI) + self.fonts.px(size + 1)

    def _tag(self, surf, text: str, color, x: float, y: float, max_width=None, size: float = 15) -> int:
        """A state tag (Blocked, Paused...): solid colour, dark text. Returns its right edge."""
        f = self.fonts
        t = f.render(text, size, theme.INK, face=DEMI)
        pad = f.px(size / 2 + 0.5)
        w = t.get_width() + 2 * pad
        if max_width is not None and w > max_width:
            w = max(f.px(24), max_width)
        r = pygame.Rect(round(x), round(y), w, t.get_height() + f.px(4))
        pygame.draw.rect(surf, color, r, border_radius=f.px(4))
        f.text(surf, text, size, theme.INK, (r.x + pad, r.centery), "midleft", face=DEMI, max_width=r.w - 2 * pad)
        return r.right
