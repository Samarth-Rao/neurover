"""Colours, type and small drawing helpers. High contrast, readable across a room, calm around the flicker."""
from __future__ import annotations

import itertools
import math
from typing import Optional

import pygame

# -- colour ------------------------------------------------------------------------------------
# The ground is the surround of the flicker boxes: keep it this dark. Colour is only ever information:
# a command (CMD_COLORS), attention / calibration (WARN), a fault (BAD), or all-good (GOOD).
BG = (11, 13, 18)
PANEL = (22, 26, 34)        # an instrument face: one section of the dashboard
PANEL_HI = (30, 35, 46)     # wells inside a face: bar tracks, lanes, unlit keys
EDGE = (52, 60, 76)         # hairlines and key-cap outlines
TEXT = (240, 243, 248)
MUTED = (150, 160, 178)     # labels
DIM = (88, 97, 114)         # footnotes, axis ticks
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

# -- type ---------------------------------------------------------------------------------------
# Two families, each with one job. DATA: DIN, the lettering of instrument panels and road signs, for every
# number, command letter and readout (tabular digits, so a changing value doesn't jitter; Bahnschrift is
# the same design on Windows). TEXT / MEDIUM / DEMI: Avenir Next for words, in sentence case.
# Sizes are design px at 1600x900, on one scale: 13 14 16 18 21 24 30 36 48 (+ the command letter).
DATA, TEXT_FACE, MEDIUM, DEMI = "data", "text", "medium", "demi"

# Candidates per face: (SysFont name, index in a .ttc collection, bold). The first one found wins.
FACES = {
    DATA: (("dinalternate", 0, True), ("bahnschrift", 0, False), ("dejavusanscondensed", 0, True),
           ("arial", 0, True)),
    TEXT_FACE: (("avenirnext", 7, False), ("segoeui", 0, False), ("helveticaneue", 0, False),
                ("dejavusans", 0, False), ("arial", 0, False)),
    MEDIUM: (("avenirnext", 5, False), ("segoeuisemibold", 0, False), ("helveticaneue", 0, False),
             ("dejavusans", 0, False), ("arial", 0, False)),
    DEMI: (("avenirnext", 2, False), ("segoeuisemibold", 0, False), ("segoeui", 0, True),
           ("dejavusans", 0, True), ("arial", 0, True)),
}
# Glyphs a face lacks (arrows, ○, ≥) are borrowed from here, on the same baseline.
GLYPH_FALLBACK = (("sfns", 0, False), ("segoeuisymbol", 0, False), ("dejavusans", 0, False),
                  ("arialunicode", 0, False))
_STRONG = {DATA, DEMI}       # borrowed glyphs are emboldened to sit with these

# Legacy faces (the scope app, and calls without ``face``).
FONT_NAMES = "segoeui,helveticaneue,helvetica,arial,dejavusans"
MONO_NAMES = "consolas,menlo,dejavusansmono,couriernew"


def cmd_color(command: Optional[str]) -> tuple[int, int, int]:
    return CMD_COLORS.get(command, NONE_COLOR) if command else NONE_COLOR


def lerp_color(a, b, t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return (round(a[0] + (b[0] - a[0]) * t), round(a[1] + (b[1] - a[1]) * t), round(a[2] + (b[2] - a[2]) * t))


_resolved: dict = {}


def _resolve(candidates) -> Optional[tuple[str, int]]:
    """(file, collection index) of the first candidate installed, or None."""
    if candidates in _resolved:
        return _resolved[candidates]
    found = None
    for name, index, bold in candidates:
        try:
            path = pygame.font.match_font(name, bold=bold)
        except Exception:
            path = None
        if not path or (index and not path.lower().endswith((".ttc", ".otc"))):
            continue
        try:
            import pygame.freetype as ft
            if not ft.get_init():
                ft.init()
            ft.Font(path, 12, font_index=index)
        except Exception:
            continue
        found = (path, index)
        break
    _resolved[candidates] = found
    return found


class _Face:
    """A pygame.font.Font look-alike on pygame.freetype at one pixel size. Unlike pygame.font it can pick
    one weight out of a font collection; glyphs the face lacks come from a fallback face on the same
    baseline. Surfaces are full line height (like pygame.font), so layouts line up across faces."""

    def __init__(self, main, fallback, px: int, strong_fallback: bool) -> None:
        self.main, self.fallback, self.px = main, fallback, px
        self._style = 0
        if strong_fallback:
            import pygame.freetype as ft
            self._style = ft.STYLE_STRONG
        faces = [main] + ([fallback] if fallback is not None else [])
        self.ascent = max(f.get_sized_ascender(px) for f in faces)
        self.descent = max(-f.get_sized_descender(px) for f in faces)
        self.height = self.ascent + self.descent
        self._missing: dict[str, bool] = {}
        self._adv: dict[tuple[int, str], float] = {}     # per-character advance, per face
        self._widths: dict[str, int] = {}

    def _runs(self, text: str) -> list[list]:
        if self.fallback is None:
            return [[self.main, text]]
        runs: list[list] = []
        for ch in text:
            miss = self._missing.get(ch)
            if miss is None:
                m = self.main.get_metrics(ch, size=self.px)
                miss = self._missing[ch] = bool(m) and m[0] is None and not ch.isspace()
            font = self.fallback if miss else self.main
            if runs and runs[-1][0] is font:
                runs[-1][1] += ch
            else:
                runs.append([font, ch])
        return runs

    def _advance(self, font, s: str) -> float:
        adv, fid, total = self._adv, id(font), 0.0
        for ch in s:
            a = adv.get((fid, ch))
            if a is None:
                m = font.get_metrics(ch, size=self.px)
                a = adv[(fid, ch)] = m[0][4] if m and m[0] else 0.0
            total += a
        return total

    def size(self, text: str) -> tuple[int, int]:
        w = self._widths.get(text)
        if w is None:
            if len(self._widths) > 4000:
                self._widths.clear()
            w = self._widths[text] = math.ceil(sum(self._advance(f, s) for f, s in self._runs(text))) + 1
        return w, self.height

    def render(self, text: str, antialias: bool, color, background=None) -> pygame.Surface:
        w, h = self.size(text)
        surf = pygame.Surface((max(1, w), h), pygame.SRCALPHA)
        if background is not None:
            surf.fill(background)
        runs = self._runs(text)
        x = 0.0
        for i, (font, s) in enumerate(runs):
            style = self._style if font is self.fallback else 0
            font.render_to(surf, (round(x), self.ascent), s, color, style=style, size=self.px)
            if i + 1 < len(runs):
                x += self._advance(font, s)
        return surf

    def get_linesize(self) -> int:
        return self.height

    def get_height(self) -> int:
        return self.height

    def get_ascent(self) -> int:
        return self.ascent

    def get_descent(self) -> int:
        return -self.descent


class Fonts:
    """Font + rendered-text cache. Sizes are in design pixels (1600x900) and scale with the window.

    ``face`` picks one of the FACES (DATA, TEXT_FACE, MEDIUM, DEMI); without it, ``bold`` / ``mono``
    pick the legacy system sans or mono (the scope app uses those)."""

    def __init__(self) -> None:
        self.scale = 1.0
        self._fonts: dict[tuple, object] = {}
        self._text: dict[tuple, pygame.Surface] = {}
        self._ft: dict[tuple, object] = {}

    def set_scale(self, scale: float) -> None:
        scale = max(0.4, round(scale, 3))
        if scale != self.scale:
            self.scale = scale
            self._text.clear()

    def px(self, v: float) -> int:
        return max(1, round(v * self.scale))

    def _freetype(self, spec: tuple[str, int]):
        f = self._ft.get(spec)
        if f is None:
            import pygame.freetype as ft
            f = self._ft[spec] = ft.Font(spec[0], 12, font_index=spec[1])
            f.origin = True
            f.kerning = False
        return f

    def font(self, size: float, bold: bool = False, mono: bool = False, face: Optional[str] = None):
        px = max(8, round(size * self.scale))
        key = (px, bold, mono, face)
        f = self._fonts.get(key)
        if f is not None:
            return f
        if face is not None:
            spec = _resolve(FACES[face])
            if spec is not None:
                fb = _resolve(GLYPH_FALLBACK)
                try:
                    f = _Face(self._freetype(spec), self._freetype(fb) if fb and fb != spec else None, px,
                              face in _STRONG)
                except Exception:
                    f = None
            if f is None:       # nothing installed: the legacy font, bold for the bold-ish faces
                return self.font(size, bold=face in _STRONG, mono=mono)
        else:
            try:
                f = pygame.font.SysFont(MONO_NAMES if mono else FONT_NAMES, px, bold=bold)
            except Exception:
                f = pygame.font.Font(None, round(px * 1.3))
        self._fonts[key] = f
        return f

    def render(self, text: str, size: float, color, bold: bool = False, mono: bool = False,
               face: Optional[str] = None) -> pygame.Surface:
        key = (text, size, tuple(color), bold, mono, face, self.scale)
        cache = self._text
        surf = cache.pop(key, None)            # least recently used first: what's on screen stays cached,
        if surf is None:                       # so there is never a frame that re-renders all the text
            if len(cache) >= 2000:
                for old in list(itertools.islice(cache, 500)):
                    del cache[old]
            surf = self.font(size, bold, mono, face).render(text, True, color)
        cache[key] = surf
        return surf

    def width(self, text: str, size: float, bold: bool = False, mono: bool = False,
              face: Optional[str] = None) -> int:
        return self.font(size, bold, mono, face).size(text)[0]

    def text(self, target: pygame.Surface, text: str, size: float, color, pos, anchor: str = "topleft",
             bold: bool = False, mono: bool = False, max_width: Optional[int] = None,
             face: Optional[str] = None) -> pygame.Rect:
        surf = self.render(text, size, color, bold, mono, face)
        if max_width is not None and surf.get_width() > max_width and len(text) > 1:
            # Truncate with an ellipsis until it fits.
            lo, hi = 0, len(text)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if self.render(text[:mid].rstrip() + "…", size, color, bold, mono, face).get_width() <= max_width:
                    lo = mid
                else:
                    hi = mid - 1
            surf = self.render(text[:lo].rstrip() + "…", size, color, bold, mono, face)
        rect = surf.get_rect(**{anchor: (round(pos[0]), round(pos[1]))})
        target.blit(surf, rect)
        return rect


def wrap(fonts: "Fonts", text: str, size: float, max_width: int, bold: bool = False, max_lines: int = 3,
         face: Optional[str] = None) -> list[str]:
    """Greedy word wrap to ``max_width`` pixels."""
    words, lines, line = text.split(), [], ""
    font = fonts.font(size, bold, face=face)
    for word in words:
        trial = f"{line} {word}".strip()
        if line and font.size(trial)[0] > max_width:
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


def face(surf: pygame.Surface, rect: pygame.Rect, fonts: "Fonts") -> None:
    """An instrument face: one dashboard section, set apart by its fill (no outline)."""
    pygame.draw.rect(surf, PANEL, rect, border_radius=fonts.px(8))


def dot(surf: pygame.Surface, color, center, radius: int) -> None:
    pygame.draw.circle(surf, color, (round(center[0]), round(center[1])), radius)


def keycap(fonts: "Fonts", surf: pygame.Surface, key: str, size: float, pos, anchor: str = "midleft") -> pygame.Rect:
    """A key drawn as a small outlined cap, for 'press this' hints."""
    f = fonts
    t = f.render(key, size, TEXT, face=DATA)
    r = pygame.Rect(0, 0, t.get_width() + f.px(10), t.get_height() + f.px(2))
    setattr(r, anchor, (round(pos[0]), round(pos[1])))
    pygame.draw.rect(surf, PANEL_HI, r, border_radius=f.px(4))
    pygame.draw.rect(surf, EDGE, r, width=1, border_radius=f.px(4))
    surf.blit(t, t.get_rect(center=(r.centerx, r.centery + 1)))
    return r
