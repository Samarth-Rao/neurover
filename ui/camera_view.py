"""The camera feed in the middle of the window, in place of the simulated arena (main.py --camera).

A new frame is turned into a surface and scaled once when it arrives; every other display frame just
blits that, so the 60 Hz flicker keeps its timing.
"""
from __future__ import annotations

from typing import Optional, Sequence

import pygame

from rover.camera import Camera

from . import theme
from .arena_view import ArenaView
from .theme import Fonts


class CameraView:
    def __init__(self, fonts: Fonts) -> None:
        self.fonts = fonts
        self._key: Optional[tuple] = None
        self._scaled: Optional[pygame.Surface] = None

    def draw(self, surf: pygame.Surface, rect: pygame.Rect, camera: Camera,
             banners: Sequence[tuple[str, tuple[int, int, int]]] = ()) -> None:
        f = self.fonts
        pygame.draw.rect(surf, (0, 0, 0), rect)
        frame, t = camera.latest()
        if frame is not None:
            key = (t, rect.size)
            if key != self._key:
                h, w = frame.shape[:2]
                img = pygame.image.frombuffer(frame.tobytes(), (w, h), "RGB")
                k = min(rect.w / w, rect.h / h)
                self._scaled = pygame.transform.scale(img, (max(1, round(w * k)), max(1, round(h * k))))
                self._key = key
            surf.blit(self._scaled, self._scaled.get_rect(center=rect.center))
        live, text = camera.status()
        if not live:
            for i, line in enumerate(theme.wrap(f, text, 22, rect.w - f.px(60), max_lines=4, face=theme.DEMI)):
                f.text(surf, line, 22, theme.WARN, (rect.centerx, rect.centery + i * f.px(28)), "midtop",
                       face=theme.DEMI)
        # Bottom-left tag: what's showing, and the key that switches it.
        lab = f.render(text, 15, theme.TEXT, face=theme.MEDIUM) if live else None
        hint = f.render("next camera", 15, theme.MUTED, face=theme.TEXT_FACE)
        cap_w = f.width("K", 15, face=theme.DATA) + f.px(10)
        inner_w = (lab.get_width() + f.px(16) if lab else 0) + cap_w + f.px(6) + hint.get_width()
        box = pygame.Rect(rect.x + f.px(12), 0, inner_w + f.px(20), hint.get_height() + f.px(10))
        box.bottom = rect.bottom - f.px(10)
        theme.panel(surf, box, fill=theme.PANEL, border=None, radius=f.px(6))
        x = box.x + f.px(10)
        if lab:
            surf.blit(lab, lab.get_rect(midleft=(x, box.centery)))
            x += lab.get_width() + f.px(16)
        cap = theme.keycap(f, surf, "K", 15, (x, box.centery))
        surf.blit(hint, hint.get_rect(midleft=(cap.right + f.px(6), box.centery)))
        self._banners(surf, rect, banners)

    def _banners(self, surf, area: pygame.Rect, banners) -> int:
        return ArenaView._banners(self, surf, area, banners, size=21)
