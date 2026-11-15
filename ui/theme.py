"""Colours, fonts and small drawing helpers. High contrast, readable across a room."""
from __future__ import annotations

from typing import Optional

import pygame

BG = (11, 13, 18)
PANEL = (22, 26, 34)
PANEL_HI = (30, 35, 46)
EDGE = (52, 60, 76)
TEXT = (240, 243, 248)
MUTED = (150, 160, 178)
DIM = (88, 97, 114)
GOOD = (52, 211, 120)
BAD = (255, 72, 88)
WARN = (255, 196, 0)
INK = (10, 12, 16)          # text drawn on top of a bright fill

CMD_COLORS = {
    "F": (52, 211, 120),    # green
    "B": (255, 152, 40),    # orange
    "L": (64, 170, 255),    # blue
    "R": (204, 120, 255),   # violet
    "S": (130, 140, 158),   # slate
}
NONE_COLOR = (44, 50, 62)

# Arena
FLOOR = (24, 28, 36)
GRID = (36, 42, 54)
WALL = (196, 204, 216)
BOX = (78, 86, 102)
BOX_EDGE = (128, 138, 156)
ROVER_BODY = (236, 240, 246)
ROVER_TRACK = (40, 44, 52)
TRAIL = (64, 200, 255)

FONT_NAMES = "segoeui,helveticaneue,helvetica,arial,dejavusans"
MONO_NAMES = "consolas,menlo,dejavusansmono,couriernew"


def cmd_color(command: Optional[str]) -> tuple[int, int, int]:
    return CMD_COLORS.get(command, NONE_COLOR) if command else NONE_COLOR


def lerp_color(a, b, t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return (round(a[0] + (b[0] - a[0]) * t), round(a[1] + (b[1] - a[1]) * t), round(a[2] + (b[2] - a[2]) * t))


class Fonts:
    """Font + rendered-text cache. Sizes are in design pixels (1600x900) and scale with the window."""

    def __init__(self) -> None:
        self.scale = 1.0
        self._fonts: dict[tuple[int, bool, bool], pygame.font.Font] = {}
        self._text: dict[tuple, pygame.Surface] = {}

    def set_scale(self, scale: float) -> None:
        scale = max(0.4, round(scale, 3))
        if scale != self.scale:
            self.scale = scale
            self._text.clear()

    def px(self, v: float) -> int:
        return max(1, round(v * self.scale))

    def font(self, size: float, bold: bool = False, mono: bool = False) -> pygame.font.Font:
        px = max(8, round(size * self.scale))
        key = (px, bold, mono)
        f = self._fonts.get(key)
        if f is None:
            try:
                f = pygame.font.SysFont(MONO_NAMES if mono else FONT_NAMES, px, bold=bold)
            except Exception:
                f = pygame.font.Font(None, round(px * 1.3))
            self._fonts[key] = f
        return f

    def render(self, text: str, size: float, color, bold: bool = False, mono: bool = False) -> pygame.Surface:
        key = (text, size, tuple(color), bold, mono, self.scale)
        surf = self._text.get(key)
        if surf is None:
            if len(self._text) > 2000:
                self._text.clear()
            surf = self.font(size, bold, mono).render(text, True, color)
            self._text[key] = surf
        return surf

    def text(self, target: pygame.Surface, text: str, size: float, color, pos, anchor: str = "topleft",
             bold: bool = False, mono: bool = False, max_width: Optional[int] = None) -> pygame.Rect:
        surf = self.render(text, size, color, bold, mono)
        if max_width is not None and surf.get_width() > max_width and len(text) > 1:
            # Truncate with an ellipsis until it fits.
            lo, hi = 0, len(text)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if self.render(text[:mid] + "…", size, color, bold, mono).get_width() <= max_width:
                    lo = mid
                else:
                    hi = mid - 1
            surf = self.render(text[:lo] + "…", size, color, bold, mono)
        rect = surf.get_rect(**{anchor: (round(pos[0]), round(pos[1]))})
        target.blit(surf, rect)
        return rect


def wrap(fonts: "Fonts", text: str, size: float, max_width: int, bold: bool = False, max_lines: int = 3) -> list[str]:
    """Greedy word wrap to ``max_width`` pixels."""
    words, lines, line = text.split(), [], ""
    for word in words:
        trial = f"{line} {word}".strip()
        if line and fonts.font(size, bold).size(trial)[0] > max_width:
            lines.append(line)
            line = word
        else:
            line = trial
    if line:
        lines.append(line)
    if len(lines) > max_lines:
        lines = lines[: max_lines - 1] + [" ".join(lines[max_lines - 1:])]
    return lines


def panel(surf: pygame.Surface, rect: pygame.Rect, fill=PANEL, border=EDGE, radius: int = 10, width: int = 1) -> None:
    pygame.draw.rect(surf, fill, rect, border_radius=radius)
    if border is not None and width > 0:
        pygame.draw.rect(surf, border, rect, width=width, border_radius=radius)


def dot(surf: pygame.Surface, color, center, radius: int) -> None:
    pygame.draw.circle(surf, color, (round(center[0]), round(center[1])), radius)
