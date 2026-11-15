"""Chase-cam view of the simulated rover: behind and above it, looking steeply down.

Same contract as ArenaView (``ChaseView(fonts).draw(surf, rect, sim, banners)``), but the
camera is locked to the rover so it always faces up the screen and the world turns around it.
World frame as in sinks/sim_sink.py (metres, +x right, +y up, heading CCW from +x) plus +z up.
Painter's algorithm: floor-level marks first, then walls/boxes as prisms sorted far to near,
then the rover last so it is never hidden.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import pygame

import commands
import config
from rover.sim import RoverSim

from . import theme
from .arena_view import ArenaView
from .theme import Fonts

# Tunables live in config.py.
from config import (  # noqa: E402
    CHASE_PITCH_DEG,
    CHASE_FOV_DEG,
    CHASE_DISTANCE_M,
    CHASE_LOOK_AHEAD_M,
    CHASE_YAW_SMOOTH_S,
    CHASE_POS_SMOOTH_S,
    CHASE_SNAP_DIST_M,
    CHASE_SNAP_DEG,
    CHASE_NEAR_M,
    CHASE_WALL_HEIGHT_M,
    CHASE_WALL_THICK_M,
    CHASE_BOX_HEIGHT_M,
    CHASE_TRACK_HEIGHT_M,
    CHASE_BODY_HEIGHT_M,
    CHASE_HALO_RADIUS_M,
    CHASE_TRAIL_WIDTH_M,
    CHASE_TRAIL_BANDS,
    CHASE_LIGHT_DIR,
)

VOID = theme.BG                         # beyond the walls
WALL_TOP = theme.WALL
WALL_SIDE = (138, 146, 162)
BOX_TOP = theme.BOX
BOX_SIDE = (58, 64, 78)
FACE_EDGE = (14, 16, 22)
ROVER_SIDE = (168, 175, 188)

_HALO_PTS = 40
_CIRCLE = [(math.cos(2 * math.pi * i / _HALO_PTS), math.sin(2 * math.pi * i / _HALO_PTS)) for i in range(_HALO_PTS)]


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _shade(base, nx: float, ny: float) -> tuple[int, int, int]:
    """Darken ``base`` for a vertical face with outward normal (nx, ny) under the fixed light."""
    lx, ly = CHASE_LIGHT_DIR
    n = math.hypot(nx, ny) or 1.0
    ln = math.hypot(lx, ly) or 1.0
    k = 0.78 + 0.22 * (nx * lx + ny * ly) / (n * ln)       # 0.56 .. 1.0
    return (round(base[0] * k), round(base[1] * k), round(base[2] * k))


def _clip_halfplane(poly, ax, ay, nx, ny):
    """Keep the part of a 2D polygon where (p - a) . n >= 0."""
    out = []
    if not poly:
        return out
    px, py = poly[-1]
    pd = (px - ax) * nx + (py - ay) * ny
    for cx, cy in poly:
        cd = (cx - ax) * nx + (cy - ay) * ny
        if (cd >= 0) != (pd >= 0):
            t = pd / (pd - cd)
            out.append((px + (cx - px) * t, py + (cy - py) * t))
        if cd >= 0:
            out.append((cx, cy))
        px, py, pd = cx, cy, cd
    return out


def _outside_regions(poly, rect: pygame.Rect) -> list[list[tuple[float, float]]]:
    """Pieces of ``rect`` outside a convex screen polygon (one per edge; they may overlap)."""
    n = len(poly)
    area2 = sum(poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1] for i in range(n))
    sign = 1.0 if area2 > 0 else -1.0
    box = [(rect.left, rect.top), (rect.right, rect.top), (rect.right, rect.bottom), (rect.left, rect.bottom)]
    out = []
    for i in range(n):
        (ax, ay), (bx, by) = poly[i], poly[(i + 1) % n]
        # Outward normal of edge a->b for this winding.
        nx, ny = (by - ay) * sign, (ax - bx) * sign
        piece = _clip_halfplane(box, ax, ay, nx, ny)
        if len(piece) >= 3:
            out.append(piece)
    return out


def sonar_color(d: float) -> tuple[int, int, int]:
    return theme.BAD if d < config.BLOCK_DISTANCE_M else theme.WARN if d < 0.4 else theme.GOOD


class ChaseCamera:
    """Perspective camera behind and above a ground pose, yawed to ``yaw`` and pitched down.

    Camera space: xc = screen right (the rover's right), yc = screen up, zc = depth along the
    view direction. ``project`` returns None for points closer than the near plane.
    """

    def __init__(self, x: float, y: float, yaw: float, view: pygame.Rect,
                 pitch_deg: Optional[float] = None, fov_deg: Optional[float] = None,
                 distance: Optional[float] = None, look_ahead: Optional[float] = None,
                 near: Optional[float] = None) -> None:
        # None -> the module tunable, read at call time so it can be changed live.
        pitch_deg = CHASE_PITCH_DEG if pitch_deg is None else pitch_deg
        fov_deg = CHASE_FOV_DEG if fov_deg is None else fov_deg
        distance = CHASE_DISTANCE_M if distance is None else distance
        look_ahead = CHASE_LOOK_AHEAD_M if look_ahead is None else look_ahead
        near = CHASE_NEAR_M if near is None else near
        self.yaw = yaw
        fx, fy = math.cos(yaw), math.sin(yaw)
        p = math.radians(pitch_deg)
        sp, cp = math.sin(p), math.cos(p)
        self.rx, self.ry = fy, -fx                         # right (horizontal)
        self.ux, self.uy, self.uz = sp * fx, sp * fy, cp   # up
        self.dx, self.dy, self.dz = cp * fx, cp * fy, -sp  # forward / view direction
        tx, ty = x + fx * look_ahead, y + fy * look_ahead
        self.px, self.py, self.pz = tx - self.dx * distance, ty - self.dy * distance, sp * distance
        self.focal = 0.5 * min(view.w, view.h) / math.tan(math.radians(fov_deg) / 2)
        self.ox, self.oy = view.centerx, view.centery
        self.near = near

    # -- point transforms ------------------------------------------------------------
    def to_cam(self, x: float, y: float, z: float = 0.0) -> tuple[float, float, float]:
        X, Y, Z = x - self.px, y - self.py, z - self.pz
        return (X * self.rx + Y * self.ry,
                X * self.ux + Y * self.uy + Z * self.uz,
                X * self.dx + Y * self.dy + Z * self.dz)

    def project_cam(self, c) -> tuple[float, float]:
        k = self.focal / c[2]
        return self.ox + c[0] * k, self.oy - c[1] * k

    def project(self, x: float, y: float, z: float = 0.0) -> Optional[tuple[float, float]]:
        c = self.to_cam(x, y, z)
        return self.project_cam(c) if c[2] >= self.near else None

    def scale_at(self, x: float, y: float, z: float = 0.0) -> float:
        """Pixels per metre (for something facing the camera) at a world point; 0 behind the camera."""
        zc = self.to_cam(x, y, z)[2]
        return self.focal / zc if zc >= self.near else 0.0

    # -- clipped primitives ------------------------------------------------------------
    def _cut(self, a, b):
        t = (self.near - a[2]) / (b[2] - a[2])
        return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, self.near)

    def polygon(self, pts) -> list[tuple[float, float]]:
        """Project a planar 3D polygon, clipped to the near plane (Sutherland-Hodgman)."""
        cam = [self.to_cam(*p) for p in pts]
        near = self.near
        if all(c[2] >= near for c in cam):
            return [self.project_cam(c) for c in cam]
        out = []
        prev = cam[-1]
        for cur in cam:
            if cur[2] >= near:
                if prev[2] < near:
                    out.append(self._cut(prev, cur))
                out.append(cur)
            elif prev[2] >= near:
                out.append(self._cut(prev, cur))
            prev = cur
        return [self.project_cam(c) for c in out] if len(out) >= 3 else []

    def segment(self, p0, p1) -> Optional[tuple[tuple[float, float], tuple[float, float]]]:
        a, b = self.to_cam(*p0), self.to_cam(*p1)
        near = self.near
        if a[2] < near and b[2] < near:
            return None
        if a[2] < near:
            a = self._cut(a, b)
        elif b[2] < near:
            b = self._cut(a, b)
        return self.project_cam(a), self.project_cam(b)

    def floor_points(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorised projection of floor points (N, 2) -> (screen (N, 2), depth (N,), in front mask)."""
        X = xy[:, 0] - self.px
        Y = xy[:, 1] - self.py
        Z = -self.pz
        xc = X * self.rx + Y * self.ry
        yc = X * self.ux + Y * self.uy + Z * self.uz
        zc = X * self.dx + Y * self.dy + Z * self.dz
        ok = zc >= self.near
        k = self.focal / np.where(ok, zc, 1.0)
        scr = np.stack((self.ox + xc * k, self.oy - yc * k), axis=1)
        return scr, zc, ok


class _Prism:
    """Axis-aligned box standing on the floor (wall segment or obstacle)."""
    __slots__ = ("x0", "y0", "x1", "y1", "h", "top", "side", "edge")

    def __init__(self, x0, y0, x1, y1, h, top, side, edge) -> None:
        self.x0, self.y0, self.x1, self.y1, self.h = x0, y0, x1, y1, h
        self.top, self.side, self.edge = top, side, edge


def _paint_before(a: _Prism, b: _Prism, cx: float, cy: float) -> Optional[bool]:
    """True if ``a`` must be painted before ``b`` (b may hide a), False for after, None if unknown.

    Any axis-aligned plane separating the two footprints decides it: whichever box is on the
    camera's side of that plane can never be hidden by the other one.
    """
    if a.x1 <= b.x0:
        return cx > (a.x1 + b.x0) / 2
    if b.x1 <= a.x0:
        return cx < (b.x1 + a.x0) / 2
    if a.y1 <= b.y0:
        return cy > (a.y1 + b.y0) / 2
    if b.y1 <= a.y0:
        return cy < (b.y1 + a.y0) / 2
    return None


def sort_far_to_near(prisms: Sequence[_Prism], cx: float, cy: float) -> list[_Prism]:
    """Painter's order for disjoint axis-aligned prisms seen from ground point (cx, cy)."""
    n = len(prisms)

    def dist(p: _Prism) -> float:
        return math.hypot(cx - min(max(cx, p.x0), p.x1), cy - min(max(cy, p.y0), p.y1))

    d = [dist(p) for p in prisms]
    after: list[list[int]] = [[] for _ in range(n)]   # after[i]: indices that must be painted after i
    indeg = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            r = _paint_before(prisms[i], prisms[j], cx, cy)
            if r is None:
                r = d[i] >= d[j]
            if r:
                after[i].append(j)
                indeg[j] += 1
            else:
                after[j].append(i)
                indeg[i] += 1
    out, left = [], set(range(n))
    while left:
        ready = [i for i in left if indeg[i] == 0] or list(left)   # (cycle: fall back to distance)
        i = max(ready, key=lambda k: d[k])
        left.discard(i)
        out.append(prisms[i])
        for j in after[i]:
            indeg[j] -= 1
    return out


class ChaseView(ArenaView):
    """Rover-centric 3D chase camera. Same public contract as ArenaView."""

    def __init__(self, fonts: Fonts) -> None:
        super().__init__(fonts)
        self.camera: Optional[ChaseCamera] = None
        self._cam_state: Optional[tuple[int, float, float, float, float]] = None  # sim id, t, x, y, yaw
        self._static_key = None
        self._static: list[_Prism] = []

    # -- camera ---------------------------------------------------------------------------
    def reset_camera(self) -> None:
        self._cam_state = None

    def _camera_pose(self, sim: RoverSim) -> tuple[float, float, float]:
        x, y, yaw, t = sim.x, sim.y, sim.heading, sim.time
        st = self._cam_state
        if st is not None and st[0] == id(sim):
            _, t0, cx, cy, cyaw = st
            dt = t - t0
            jump = (dt < 0 or dt > 0.5 or math.hypot(x - cx, y - cy) > CHASE_SNAP_DIST_M
                    or abs(_wrap(yaw - cyaw)) > math.radians(CHASE_SNAP_DEG))
            if not jump:
                ap = 1.0 - math.exp(-dt / CHASE_POS_SMOOTH_S) if CHASE_POS_SMOOTH_S > 0 else 1.0
                ay = 1.0 - math.exp(-dt / CHASE_YAW_SMOOTH_S) if CHASE_YAW_SMOOTH_S > 0 else 1.0
                x, y = cx + (x - cx) * ap, cy + (y - cy) * ap
                yaw = _wrap(cyaw + _wrap(yaw - cyaw) * ay)
        self._cam_state = (id(sim), t, x, y, yaw)
        return x, y, yaw

    # -- main entry point -------------------------------------------------------------------
    def draw(self, surf: pygame.Surface, rect: pygame.Rect, sim: Optional[RoverSim],
             banners: Sequence[tuple[str, tuple[int, int, int]]] = ()) -> None:
        if sim is None:
            self.reset_camera()
            self.camera = None
            super().draw(surf, rect, None, banners)
            return
        f = self.fonts
        compact = rect.h < f.px(420)     # short view (centre of the SSVEP layout): one-line readouts
        pad = f.px(24)                   # (kept off the flicker boxes at the sides)
        pad_y = f.px(8) if compact else pad
        readout_h = f.px(28) if compact else f.px(64)
        gap = f.px(6) if compact else f.px(10)
        self._readouts(surf, pygame.Rect(rect.x + pad, rect.y + pad_y, rect.w - 2 * pad, readout_h), sim, compact)
        view = pygame.Rect(rect.x + pad, rect.y + pad_y + readout_h + gap,
                           rect.w - 2 * pad, rect.h - 2 * pad_y - readout_h - gap)

        bl = self.sim_banners(sim, banners)
        if view.w < 40 or view.h < 40:
            self._banners(surf, rect, bl)
            return

        cam = self.camera = ChaseCamera(*self._camera_pose(sim), view)
        old_clip = surf.get_clip()
        surf.set_clip(view.clip(old_clip))
        try:
            self._scene(surf, view, sim, cam)
        finally:
            surf.set_clip(old_clip)
        pygame.draw.rect(surf, theme.EDGE, view, 1)
        self._compass(surf, view, cam.yaw)
        self._banners(surf, view, bl)

    # -- scene ---------------------------------------------------------------------------------
    def _scene(self, surf, view: pygame.Rect, sim: RoverSim, cam: ChaseCamera) -> None:
        f = self.fonts
        arena = sim.arena
        W, H = arena.width, arena.height
        # The floor usually covers most of the view: fill the rect (cheap) and paint only the void
        # outside each floor edge, instead of a void fill plus a big (slow) floor polygon.
        floor = cam.polygon(((0, 0, 0), (W, 0, 0), (W, H, 0), (0, H, 0)))
        if floor:
            surf.fill(theme.FLOOR, view)
            for void in _outside_regions(floor, view):
                pygame.draw.polygon(surf, VOID, void)
        else:
            surf.fill(VOID, view)

        active = commands.STOP if (sim.blocked or sim.link_lost) else sim.cmd
        acol = theme.cmd_color(active)
        halo = [(sim.x + c * CHASE_HALO_RADIUS_M, sim.y + s * CHASE_HALO_RADIUS_M) for c, s in _CIRCLE]
        for ax, ay, nx, ny in ((0, 0, 1, 0), (0, 0, 0, 1), (W, H, -1, 0), (W, H, 0, -1)):
            halo = _clip_halfplane(halo, ax, ay, nx, ny)      # keep it on the floor, not under walls
        halo_scr = cam.polygon([(x, y, 0.0) for x, y in halo]) if len(halo) >= 3 else []
        if halo_scr:   # (opaque pre-blended fills stand in for alpha: no per-frame SRCALPHA layers)
            pygame.draw.polygon(surf, theme.lerp_color(theme.FLOOR, acol, 0.16), halo_scr)

        scol = sonar_color(sim.distance)
        ox, oy = sim.sensor_origin
        cone = [(ox, oy, 0.0)] + [(ox + math.cos(a) * r, oy + math.sin(a) * r, 0.0) for a, r in sim.rays]
        cone_scr = cam.polygon(cone) if len(cone) >= 3 else []
        if cone_scr:
            pygame.draw.polygon(surf, theme.lerp_color(theme.FLOOR, scol, 0.2), cone_scr)

        self._grid3d(surf, cam, W, H)
        self._trail3d(surf, sim, cam)

        mid_a, mid_r = sim.rays[len(sim.rays) // 2]
        seg = cam.segment((ox, oy, 0.0), (ox + math.cos(mid_a) * mid_r, oy + math.sin(mid_a) * mid_r, 0.0))
        if seg:
            pygame.draw.line(surf, scol, seg[0], seg[1], max(2, f.px(2)))
        if halo_scr:
            pygame.draw.polygon(surf, theme.lerp_color(theme.FLOOR, acol, 0.6), halo_scr, max(2, f.px(2)))

        cpx, cpy = cam.px, cam.py
        for p in sort_far_to_near(self._prisms(arena), cpx, cpy):
            self._prism(surf, cam, p)

        self._rover3d(surf, sim, cam, acol)
        self._sonar_label(surf, view, sim, cam, scol)

    def _prisms(self, arena) -> list[_Prism]:
        key = (id(arena), arena.width, arena.height, tuple(arena.boxes))
        if key != self._static_key:
            W, H, t, hw = arena.width, arena.height, CHASE_WALL_THICK_M, CHASE_WALL_HEIGHT_M
            walls = [(-t, -t, W + t, 0), (-t, H, W + t, H + t), (-t, 0, 0, H), (W, 0, W + t, H)]
            self._static = ([_Prism(*w, hw, WALL_TOP, WALL_SIDE, theme.MUTED) for w in walls]
                            + [_Prism(b.x0, b.y0, b.x1, b.y1, CHASE_BOX_HEIGHT_M, BOX_TOP, BOX_SIDE,
                                      theme.BOX_EDGE) for b in arena.boxes])
            self._static_key = key
        return self._static

    def _prism(self, surf, cam: ChaseCamera, p: _Prism) -> None:
        corners = ((p.x0, p.y0), (p.x1, p.y0), (p.x1, p.y1), (p.x0, p.y1))
        self._extrude(surf, cam, corners, 0.0, p.h, p.top, p.side, p.edge)

    def _extrude(self, surf, cam: ChaseCamera, corners, z0: float, z1: float, top, side, edge) -> None:
        """Draw a vertical prism over the CCW footprint ``corners``: visible side faces, then the top."""
        cx, cy = cam.px, cam.py
        n = len(corners)
        for i in range(n):
            ax, ay = corners[i]
            bx, by = corners[(i + 1) % n]
            nx, ny = by - ay, ax - bx            # outward normal of a CCW edge
            if nx * (cx - ax) + ny * (cy - ay) <= 0:
                continue                         # back face
            pts = cam.polygon(((ax, ay, z0), (bx, by, z0), (bx, by, z1), (ax, ay, z1)))
            if pts:
                pygame.draw.polygon(surf, _shade(side, nx, ny), pts)
                pygame.draw.aalines(surf, FACE_EDGE, True, pts)
        if cam.pz > z1:
            pts = cam.polygon([(x, y, z1) for x, y in corners])
            if pts:
                pygame.draw.polygon(surf, top, pts)
                pygame.draw.aalines(surf, edge, True, pts)

    def _grid3d(self, surf, cam: ChaseCamera, W: float, H: float) -> None:
        step = 0.5
        lines = [((i * step, 0, 0), (i * step, H, 0), i % 2 == 0) for i in range(1, int(W / step + 1e-9) + 1)
                 if i * step < W - 1e-9]
        lines += [((0, j * step, 0), (W, j * step, 0), j % 2 == 0) for j in range(1, int(H / step + 1e-9) + 1)
                  if j * step < H - 1e-9]
        for p0, p1, major in lines:
            seg = cam.segment(p0, p1)
            if seg:
                if major:
                    pygame.draw.line(surf, theme.GRID, seg[0], seg[1], 2)
                else:
                    pygame.draw.aaline(surf, theme.GRID, seg[0], seg[1])

    def _trail3d(self, surf, sim: RoverSim, cam: ChaseCamera) -> None:
        """Fading trail as one filled floor strip per colour band (thick pygame polylines leave gaps)."""
        if len(sim.trail) < 2:
            return
        arr = np.asarray(sim.trail, dtype=float)
        xy = arr[:, :2]
        tangent = np.empty_like(xy)
        tangent[1:-1] = xy[2:] - xy[:-2]
        tangent[0], tangent[-1] = xy[1] - xy[0], xy[-1] - xy[-2]
        norm = np.hypot(tangent[:, 0], tangent[:, 1])
        tangent /= np.maximum(norm, 1e-9)[:, None]
        offset = np.stack((-tangent[:, 1], tangent[:, 0]), axis=1) * (CHASE_TRAIL_WIDTH_M / 2)
        left, _, ok_l = cam.floor_points(xy + offset)
        right, _, ok_r = cam.floor_points(xy - offset)
        ok = ok_l & ok_r
        nb = max(1, CHASE_TRAIL_BANDS)
        age = np.clip((sim.time - arr[1:, 2]) / config.TRAIL_S, 0.0, 0.999)
        key = np.where(ok[:-1] & ok[1:], (age * nb).astype(int), -1)      # one entry per segment
        breaks = np.flatnonzero(key[1:] != key[:-1]) + 1
        starts = np.concatenate(([0], breaks)).tolist()
        ends = np.concatenate((breaks, [len(key)])).tolist()
        left_l, right_l = left.tolist(), right.tolist()
        for s, e in zip(starts, ends):
            k = int(key[s])
            if k < 0:
                continue
            col = theme.lerp_color(theme.TRAIL, theme.FLOOR, ((k + 0.5) / nb) ** 0.8)
            pygame.draw.polygon(surf, col, left_l[s:e + 1] + right_l[s:e + 1][::-1])

    def _rover3d(self, surf, sim: RoverSim, cam: ChaseCamera, acol) -> None:
        c, s = math.cos(sim.heading), math.sin(sim.heading)
        X, Y = sim.x, sim.y

        def world(points):
            return [(X + px * c - py * s, Y + px * s + py * c) for px, py in points]

        L, Wd = config.ROVER_LENGTH_M, config.ROVER_WIDTH_M
        tw = 0.024
        hl, hw = L / 2, Wd / 2
        th, bh = CHASE_TRACK_HEIGHT_M, CHASE_BODY_HEIGHT_M
        # Footprints are CCW in the rover frame (x forward, y left).
        left_track = world([(-hl, hw - tw), (hl, hw - tw), (hl, hw), (-hl, hw)])
        right_track = world([(-hl, -hw), (hl, -hw), (hl, -hw + tw), (-hl, -hw + tw)])
        body = world([(-hl * 0.92, -hw + tw), (hl * 0.92, -hw + tw), (hl * 0.92, hw - tw), (-hl * 0.92, hw - tw)])
        # Soft contact shadow so the body reads as standing on the floor.
        shadow = cam.polygon([(x, y, 0.0) for x, y in world(
            [(-hl - 0.01, -hw - 0.012), (hl + 0.01, -hw - 0.012), (hl + 0.01, hw + 0.012), (-hl - 0.01, hw + 0.012)])])
        if shadow:
            pygame.draw.polygon(surf, theme.lerp_color(theme.FLOOR, (0, 0, 0), 0.45), shadow)
        for tr in (left_track, right_track):
            self._extrude(surf, cam, tr, 0.0, th, theme.ROVER_TRACK, (58, 63, 74), theme.DIM)
        # Body sits between the tracks; starting its sides at track height keeps tracks-then-body
        # a correct painter's order from any camera position.
        self._extrude(surf, cam, body, th, bh, theme.ROVER_BODY, ROVER_SIDE, (150, 158, 172))
        arrow = world([(hl * 0.72, 0.0), (-hl * 0.38, hw * 0.52), (-hl * 0.12, 0.0), (-hl * 0.38, -hw * 0.52)])
        pts = cam.polygon([(x, y, bh + 1e-4) for x, y in arrow])
        if pts:
            pygame.draw.polygon(surf, acol, pts)
            pygame.draw.aalines(surf, theme.lerp_color(acol, (0, 0, 0), 0.35), True, pts)
        # Ultrasonic "eyes" along the front edge.
        r = max(2, round(0.012 * cam.scale_at(X, Y, bh)))
        for ex, ey in world([(hl * 0.92, 0.022), (hl * 0.92, -0.022)]):
            p = cam.project(ex, ey, bh)
            if p:
                theme.dot(surf, (60, 66, 78), p, r)
                pygame.draw.circle(surf, (200, 206, 216), (round(p[0]), round(p[1])), r, 1)

    def _sonar_label(self, surf, view: pygame.Rect, sim: RoverSim, cam: ChaseCamera, color) -> None:
        a, r = min(sim.rays, key=lambda ar: ar[1])
        if r >= config.ULTRASONIC_MAX_RANGE_M - 1e-6:
            return
        ox, oy = sim.sensor_origin
        hit = cam.project(ox + math.cos(a) * r, oy + math.sin(a) * r, 0.0)
        if hit is None:
            return
        f = self.fonts
        label = f.render(f"{r * 100:.0f} cm", 22, color, face=theme.DATA)
        inner = view.inflate(-2 * f.px(10), -2 * f.px(10))
        if inner.collidepoint(hit):
            theme.dot(surf, color, hit, f.px(6))
            pygame.draw.circle(surf, theme.INK, (round(hit[0]), round(hit[1])), f.px(6), 1)
            lr = label.get_rect(midleft=(round(hit[0] + f.px(14)), round(hit[1])))
        else:   # hit is beyond the edge of the view: pin the reading just inside it
            lr = label.get_rect(midleft=(round(hit[0] + f.px(14)), round(hit[1])))
            lr.clamp_ip(inner)
        theme.panel(surf, lr.inflate(f.px(12), f.px(4)), fill=theme.BG, border=None, radius=f.px(6))
        surf.blit(label, lr)

    def _compass(self, surf, view: pygame.Rect, yaw: float) -> None:
        """World-north (+y) needle, so viewers can see the world turning under the rover."""
        f = self.fonts
        rad = max(12, min(f.px(24), round(min(view.w, view.h) * 0.06)))
        ring = rad + f.px(11)              # the "N" orbits on this radius
        legend = f.render("Grid 0.5 m", 15, theme.MUTED, face=theme.MEDIUM) if view.w >= 560 else None
        half = ring + f.px(11)
        box = pygame.Rect(0, 0, 2 * half + (legend.get_width() + f.px(14) if legend else 0), 2 * half)
        box.bottomleft = (view.left + f.px(10), view.bottom - f.px(10))
        theme.panel(surf, box, fill=theme.PANEL, border=None, radius=f.px(8))
        cx, cy = box.left + half, box.centery
        if legend:
            surf.blit(legend, legend.get_rect(midleft=(cx + half + f.px(4), cy)))
        pygame.draw.circle(surf, theme.BG, (cx, cy), rad)
        pygame.draw.circle(surf, theme.EDGE, (cx, cy), rad, max(1, f.px(2)))
        nx, ny = -math.cos(yaw), -math.sin(yaw)   # world +y in screen coordinates (y down)
        px, py = -ny, nx
        tip = (cx + nx * rad * 0.78, cy + ny * rad * 0.78)
        tail = (cx - nx * rad * 0.78, cy - ny * rad * 0.78)
        w = rad * 0.26
        pygame.draw.polygon(surf, theme.BAD, [tip, (cx + px * w, cy + py * w), (cx - px * w, cy - py * w)])
        pygame.draw.polygon(surf, theme.MUTED, [tail, (cx + px * w, cy + py * w), (cx - px * w, cy - py * w)])
        f.text(surf, "N", 16, theme.TEXT, (cx + nx * ring, cy + ny * ring + 1), "center", face=theme.DATA)
