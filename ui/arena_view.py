"""Draws the top-down arena: walls, boxes, fading trail, ultrasonic cone and the rover."""
from __future__ import annotations

import math
from typing import Optional, Sequence

import pygame

import commands
import config
from rover.sim import RoverSim

from . import theme
from .theme import Fonts


class ArenaView:
    def __init__(self, fonts: Fonts) -> None:
        self.fonts = fonts

    def draw(self, surf: pygame.Surface, rect: pygame.Rect, sim: Optional[RoverSim],
             banners: Sequence[tuple[str, tuple[int, int, int]]] = ()) -> None:
        f = self.fonts
        pad = f.px(24)
        if sim is None:
            theme.panel(surf, rect.inflate(-2 * pad, -2 * pad), fill=theme.PANEL, border=None, radius=f.px(8))
            f.text(surf, "Simulation off", 36, theme.MUTED, rect.center, "midbottom", face=theme.DEMI,
                   max_width=rect.w - 4 * pad)
            f.text(surf, "--no-sim: commands go to the serial rover only", 21, theme.DIM,
                   (rect.centerx, rect.centery + f.px(8)), "midtop", face=theme.TEXT_FACE, max_width=rect.w - 4 * pad)
            self._banners(surf, rect, banners)
            return

        if rect.h < f.px(420):  # short view (centre of the SSVEP layout): the arena gets all the height
            pad = f.px(8)
            avail = rect.inflate(-2 * pad, -2 * pad)
        else:
            readout_h = f.px(64)
            self._readouts(surf, pygame.Rect(rect.x + pad, rect.y + pad, rect.w - 2 * pad, readout_h), sim)
            avail = pygame.Rect(rect.x + pad, rect.y + pad + readout_h + f.px(10),
                                rect.w - 2 * pad, rect.h - 2 * pad - readout_h - f.px(10))
        W, H = sim.arena.width, sim.arena.height
        scale = min(avail.w / W, avail.h / H)
        ox = avail.x + (avail.w - W * scale) / 2
        oy = avail.y + (avail.h - H * scale) / 2

        def to_screen(x: float, y: float) -> tuple[float, float]:
            return ox + x * scale, oy + (H - y) * scale

        floor = pygame.Rect(round(ox), round(oy), round(W * scale), round(H * scale))
        pygame.draw.rect(surf, theme.FLOOR, floor)
        self._grid(surf, floor, W, H, scale, to_screen)

        for b in sim.arena.boxes:
            x0, y0 = to_screen(b.x0, b.y1)
            r = pygame.Rect(round(x0), round(y0), round((b.x1 - b.x0) * scale), round((b.y1 - b.y0) * scale))
            pygame.draw.rect(surf, theme.BOX, r, border_radius=f.px(3))
            pygame.draw.rect(surf, theme.BOX_EDGE, r, width=max(1, f.px(2)), border_radius=f.px(3))

        pygame.draw.rect(surf, theme.WALL, floor.inflate(f.px(8), f.px(8)), width=f.px(5), border_radius=f.px(4))

        self._trail(surf, sim, to_screen, scale)
        self._sonar(surf, sim, to_screen)
        self._rover(surf, sim, to_screen, scale)
        self._scale_bar(surf, floor, scale)

        # Banners over the top of the floor, but as wide as the view allows (the floor is small in the
        # flicker layout, and a wrapped warning would hide it).
        self._banners(surf, pygame.Rect(rect.x + pad, floor.top, rect.w - 2 * pad, floor.h),
                      self.sim_banners(sim, banners))

    @staticmethod
    def sim_banners(sim: RoverSim, banners) -> list:
        bl = list(banners)
        if sim.blocked:
            bl.insert(0, (f"Blocked: obstacle {sim.distance * 100:.0f} cm ahead", theme.BAD))
        elif sim.link_lost:
            bl.insert(0, ("No signal: rover stopped (link watchdog)", theme.WARN))
        return bl

    # -- pieces ------------------------------------------------------------------------
    def _readouts(self, surf, rect: pygame.Rect, sim: RoverSim, compact: bool = False) -> None:
        """Sonar, speed, turn rate and heading: a label over a DIN value, or one line each when ``compact``."""
        f = self.fonts
        d = sim.distance
        if d >= config.ULTRASONIC_MAX_RANGE_M - 1e-6:
            sonar, col = f"> {config.ULTRASONIC_MAX_RANGE_M:.1f} m", theme.TEXT
        else:
            sonar = f"{d * 100:.0f} cm"
            col = theme.BAD if d < config.BLOCK_DISTANCE_M else theme.WARN if d < 0.4 else theme.TEXT
        items = [
            ("Sonar", sonar, col),
            ("Speed", f"{sim.speed:+.2f} m/s", theme.TEXT),
            ("Turn", f"{math.degrees(sim.turn_rate):+.0f}°/s", theme.TEXT),
            ("Heading", f"{sim.heading_deg % 360:.0f}°", theme.TEXT),
        ]
        if rect.w < f.px(600 if not compact else 460):  # small view: sonar + speed only
            items = items[:2]
        cw = rect.w / len(items)
        for i, (label, value, color) in enumerate(items):
            x = rect.x + i * cw
            if compact:
                lab = f.text(surf, label, 15, theme.MUTED, (x, rect.centery), "midleft", face=theme.MEDIUM,
                             max_width=round(cw - f.px(8)))
                if lab.right + f.px(8) < x + cw - f.px(30):
                    f.text(surf, value, 21, color, (lab.right + f.px(8), rect.centery + 1), "midleft",
                           face=theme.DATA, max_width=round(x + cw - f.px(8) - lab.right - f.px(8)))
                continue
            f.text(surf, label, 15, theme.MUTED, (x, rect.y), face=theme.MEDIUM, max_width=round(cw - f.px(8)))
            f.text(surf, value, 32, color, (x, rect.y + f.px(20)), face=theme.DATA, max_width=round(cw - f.px(8)))

    def _grid(self, surf, floor, W, H, scale, to_screen) -> None:
        step = 0.5
        for i in range(1, int(W / step)):
            x = to_screen(i * step, 0)[0]
            pygame.draw.line(surf, theme.GRID, (x, floor.top), (x, floor.bottom), 2 if i % 2 == 0 else 1)
        for j in range(1, int(H / step)):
            y = to_screen(0, j * step)[1]
            pygame.draw.line(surf, theme.GRID, (floor.left, y), (floor.right, y), 2 if j % 2 == 0 else 1)

    def _trail(self, surf, sim: RoverSim, to_screen, scale) -> None:
        pts = sim.trail
        if len(pts) < 2:
            return
        width = max(2, round(0.018 * scale))
        prev = None
        for x, y, t in pts:
            p = to_screen(x, y)
            if prev is not None:
                age = (sim.time - t) / config.TRAIL_S
                pygame.draw.line(surf, theme.lerp_color(theme.TRAIL, theme.FLOOR, age ** 0.8), prev, p, width)
            prev = p

    def _sonar(self, surf, sim: RoverSim, to_screen) -> None:
        ox, oy = sim.sensor_origin
        origin = to_screen(ox, oy)
        d = sim.distance
        color = theme.BAD if d < config.BLOCK_DISTANCE_M else theme.WARN if d < 0.4 else theme.GOOD
        pts = [origin] + [to_screen(ox + math.cos(a) * r, oy + math.sin(a) * r) for a, r in sim.rays]
        if len(pts) >= 3:
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            bx, by = math.floor(min(xs)), math.floor(min(ys))
            bw, bh = math.ceil(max(xs)) - bx + 2, math.ceil(max(ys)) - by + 2
            if bw > 0 and bh > 0:
                layer = pygame.Surface((bw, bh), pygame.SRCALPHA)
                pygame.draw.polygon(layer, (*color, 46), [(px - bx, py - by) for px, py in pts])
                surf.blit(layer, (bx, by))
        # Centre ray + nearest hit.
        mid = sim.rays[len(sim.rays) // 2]
        end = to_screen(ox + math.cos(mid[0]) * mid[1], oy + math.sin(mid[0]) * mid[1])
        pygame.draw.line(surf, color, origin, end, max(2, self.fonts.px(2)))
        a, r = min(sim.rays, key=lambda ar: ar[1])
        hit = to_screen(ox + math.cos(a) * r, oy + math.sin(a) * r)
        if r < config.ULTRASONIC_MAX_RANGE_M - 1e-6:
            theme.dot(surf, color, hit, self.fonts.px(6))
            self.fonts.text(surf, f"{r * 100:.0f} cm", 22, color, (hit[0] + self.fonts.px(10), hit[1] + 1),
                            "midleft", face=theme.DATA)

    def _rover(self, surf, sim: RoverSim, to_screen, scale) -> None:
        c, s = math.cos(sim.heading), math.sin(sim.heading)

        def poly(points):
            return [to_screen(sim.x + px * c - py * s, sim.y + px * s + py * c) for px, py in points]

        # Halo so the (true-to-scale, small) rover is easy to spot from across a room.
        active = commands.STOP if (sim.blocked or sim.link_lost) else sim.cmd
        centre = to_screen(sim.x, sim.y)
        hr = max(12, round(0.17 * scale))
        halo = pygame.Surface((hr * 2 + 2, hr * 2 + 2), pygame.SRCALPHA)
        pygame.draw.circle(halo, (*theme.cmd_color(active), 40), (hr + 1, hr + 1), hr)
        pygame.draw.circle(halo, (*theme.cmd_color(active), 150), (hr + 1, hr + 1), hr, max(2, self.fonts.px(2)))
        surf.blit(halo, (round(centre[0]) - hr - 1, round(centre[1]) - hr - 1))

        L, Wd = config.ROVER_LENGTH_M, config.ROVER_WIDTH_M
        tw = 0.024
        hl, hw = L / 2, Wd / 2
        body = [(hl, hw - tw), (-hl, hw - tw), (-hl, -hw + tw), (hl, -hw + tw)]
        left_track = [(hl, hw), (-hl, hw), (-hl, hw - tw), (hl, hw - tw)]
        right_track = [(hl, -hw + tw), (-hl, -hw + tw), (-hl, -hw), (hl, -hw)]
        for tr in (left_track, right_track):
            pygame.draw.polygon(surf, theme.ROVER_TRACK, poly(tr))
            pygame.draw.polygon(surf, theme.DIM, poly(tr), 1)
        pygame.draw.polygon(surf, theme.ROVER_BODY, poly(body))
        # Direction arrow tinted with the command the rover is executing.
        arrow = [(hl * 0.75, 0.0), (-hl * 0.35, hw * 0.55), (-hl * 0.1, 0.0), (-hl * 0.35, -hw * 0.55)]
        pygame.draw.polygon(surf, theme.cmd_color(active), poly(arrow))
        # Two "eyes" of the ultrasonic sensor at the front.
        for ey in (0.022, -0.022):
            p = poly([(hl, ey)])[0]
            theme.dot(surf, (60, 66, 78), p, max(2, round(0.013 * scale)))
            pygame.draw.circle(surf, (200, 206, 216), (round(p[0]), round(p[1])), max(2, round(0.013 * scale)), 1)

    def _scale_bar(self, surf, floor: pygame.Rect, scale: float) -> None:
        f = self.fonts
        x0, y = floor.left + f.px(14), floor.bottom - f.px(16)
        x1 = x0 + scale  # 1 metre
        pygame.draw.line(surf, theme.MUTED, (x0, y), (x1, y), 2)
        for x in (x0, x1):
            pygame.draw.line(surf, theme.MUTED, (x, y - f.px(5)), (x, y + f.px(5)), 2)
        f.text(surf, "1 m", 16, theme.MUTED, ((x0 + x1) / 2, y - f.px(5)), "midbottom", face=theme.DATA)

    def _banners(self, surf, area: pygame.Rect, banners, size: float = 24) -> None:
        """Problems, one solid band each (dark text on the colour), at the top of ``area``."""
        f = self.fonts
        y = area.top + f.px(12)
        max_w = min(area.w - f.px(24), f.px(900))
        for text, color in banners:
            sz = size if f.width(text, size, face=theme.DEMI) <= max_w - f.px(36) else size * 0.85
            lines = theme.wrap(f, text, sz, max_w - f.px(36), face=theme.DEMI)
            line_h = f.px(sz * 1.25)
            w = min(max_w, max(f.width(line, sz, face=theme.DEMI) for line in lines) + f.px(36))
            r = pygame.Rect(0, 0, w, line_h * len(lines) + f.px(14))
            r.midtop = (area.centerx, y)
            pygame.draw.rect(surf, color, r, border_radius=f.px(6))
            for i, line in enumerate(lines):
                f.text(surf, line, sz, theme.INK, (r.centerx, r.y + f.px(7) + i * line_h + (line_h - f.font(
                    sz, face=theme.DEMI).get_height()) // 2), "midtop", face=theme.DEMI, max_width=r.w - f.px(20))
            y = r.bottom + f.px(8)
