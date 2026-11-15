"""Training-mode panel, drawn over the arena while the rover is paused."""
from __future__ import annotations

import pygame

import commands
import config
from sources.signal_source import SignalSource

from . import theme
from .theme import Fonts


class TrainingPanel:
    def __init__(self, fonts: Fonts) -> None:
        self.fonts = fonts

    def draw(self, surf: pygame.Surface, area: pygame.Rect, source: SignalSource, clear_armed: bool) -> None:
        f = self.fonts
        pad = f.px(22)
        row_h = f.px(54)
        n_rows = len(source.classes)
        w = area.w - 2 * f.px(16)
        h = min(area.h - 2 * f.px(16), pad * 2 + f.px(96) + row_h * n_rows + f.px(80))
        r = pygame.Rect(0, 0, w, h)
        r.midtop = (area.centerx, area.y + f.px(16))
        pygame.draw.rect(surf, theme.PANEL, r, border_radius=f.px(14))
        pygame.draw.rect(surf, theme.WARN, r, width=f.px(3), border_radius=f.px(14))

        x, y = r.x + pad, r.y + pad
        title = f.text(surf, "TRAINING MODE", 36, theme.TEXT, (x, y), bold=True)
        self._pill(surf, "ROVER PAUSED", theme.WARN, r.right - pad, title.centery)
        y = title.bottom + f.px(6)
        f.text(surf, "Hold a number key while making the gesture.", 21, theme.MUTED, (x, y), max_width=r.w - 2 * pad)
        y += f.px(36)

        counts = source.trainer.data.counts()
        hints = getattr(source, "gesture_hints", {})
        keys = {g: k for k, g in getattr(source, "label_keys", config.GESTURE_KEYS).items()}
        rows = list(source.classes)
        target = config.TRAIN_TARGET_PER_CLASS
        bar_x = x + f.px(330)
        bar_w = r.right - pad - bar_x - f.px(90)
        for g in rows:
            cy = y + row_h / 2
            rec = source.recording == g
            key_r = pygame.Rect(x, 0, f.px(44), f.px(44))
            key_r.centery = round(cy)
            pygame.draw.rect(surf, theme.BAD if rec else theme.PANEL_HI, key_r, border_radius=f.px(8))
            pygame.draw.rect(surf, theme.EDGE, key_r, width=2, border_radius=f.px(8))
            f.text(surf, str(keys.get(g, "?")), 28, theme.INK if rec else theme.TEXT, key_r.center, "center",
                   bold=True)
            name = getattr(source, "class_names", config.GESTURE_NAMES).get(g, g)
            nr = f.text(surf, name, 28, theme.TEXT, (key_r.right + f.px(14), cy), "midleft", bold=True)
            if g in hints:
                f.text(surf, hints[g], 24, theme.MUTED, (nr.right + f.px(10), cy), "midleft", bold=True)
            cmd = commands.from_label(g, source.command_map)
            chip = pygame.Rect(0, 0, f.px(34), f.px(30))
            chip.midright = (bar_x - f.px(14), round(cy))
            pygame.draw.rect(surf, theme.cmd_color(cmd), chip, border_radius=f.px(5))
            f.text(surf, cmd or "?", 21, theme.INK, chip.center, "center", bold=True)

            track = pygame.Rect(bar_x, round(cy - f.px(9)), bar_w, f.px(18))
            pygame.draw.rect(surf, theme.PANEL_HI, track, border_radius=f.px(5))
            n = counts.get(g, 0)
            if n:
                fill = track.copy()
                fill.w = max(3, round(track.w * min(1.0, n / target)))
                pygame.draw.rect(surf, theme.GOOD if n >= target else theme.cmd_color(cmd), fill,
                                 border_radius=f.px(5))
            f.text(surf, f"{n:,}", 26, theme.TEXT, (r.right - pad, cy), "midright", bold=True)
            if rec:
                settling = source.recorded_this_hold == 0
                label = "settling…" if settling else "● REC"
                f.text(surf, label, 20, theme.WARN if settling else theme.BAD,
                       (track.centerx, track.top - f.px(2)), "midbottom", bold=True)
            y += row_h

        y += f.px(10)
        tr = source.trainer
        if tr.clf.is_fitted and tr.report is not None:
            status = f"Model {tr.report.describe()}"
            if tr.dirty:
                status += "  ·  new data: press Enter to refit"
            col = theme.WARN if tr.dirty else theme.GOOD
        elif len(tr.data):
            status, col = "Model not fitted yet: press Enter", theme.WARN
        else:
            status, col = f"No data yet: aim for ~{target} windows per gesture (~{target * config.HOP_S:.0f} s held)", theme.MUTED
        f.text(surf, status, 22, col, (x, y), bold=True, max_width=r.w - 2 * pad)
        y += f.px(34)
        if clear_armed:
            f.text(surf, "Press X again to clear ALL training data", 22, theme.BAD, (x, y), bold=True)
        else:
            f.text(surf, "Enter fit  ·  Ctrl+S save  ·  X clear  ·  T exit training", 21, theme.MUTED, (x, y),
                   max_width=r.w - 2 * pad)

    def _pill(self, surf, text, color, right, cy) -> None:
        f = self.fonts
        t = f.render(text, 20, theme.INK, bold=True)
        pr = pygame.Rect(0, 0, t.get_width() + f.px(22), t.get_height() + f.px(8))
        pr.midright = (round(right), round(cy))
        pygame.draw.rect(surf, color, pr, border_radius=pr.h // 2)
        surf.blit(t, t.get_rect(center=pr.center))
