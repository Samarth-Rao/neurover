"""Simulated rover: differential-drive (tank-style) physics in a walled arena.

World frame: metres, origin at the arena's bottom-left corner, +x right, +y up,
heading in radians counter-clockwise from +x (so a LEFT turn increases heading).
This module has no pygame dependency; ui/arena_view.py draws it.

The sim behaves like the real rover firmware is expected to:
  * F/B drive both tracks at +/-ROVER_SPEED_MPS; L/R spin in place; S brakes.
  * Forward ultrasonic sensor: F is refused (treated as S) when anything is
    closer than BLOCK_DISTANCE_M ahead.
  * Link watchdog: if no command arrives for ROVER_LINK_TIMEOUT_S, stop.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Optional

import commands
import config

from .base import Sink, SinkStatus

_EPS = 1e-12


@dataclass(frozen=True)
class Box:
    x0: float
    y0: float
    x1: float
    y1: float


class Arena:
    def __init__(self, width: float = config.ARENA_W_M, height: float = config.ARENA_H_M,
                 boxes: Iterable[tuple[float, float, float, float]] = config.ARENA_BOXES) -> None:
        self.width = width
        self.height = height
        self.boxes = [Box(*b) for b in boxes]

    def circle_hits(self, x: float, y: float, r: float) -> bool:
        if x - r < 0 or y - r < 0 or x + r > self.width or y + r > self.height:
            return True
        for b in self.boxes:
            cx = min(max(x, b.x0), b.x1)
            cy = min(max(y, b.y0), b.y1)
            if (x - cx) ** 2 + (y - cy) ** 2 < r * r:
                return True
        return False

    def raycast(self, ox: float, oy: float, angle: float, max_range: float) -> float:
        """Distance from (ox, oy) along ``angle`` to the first wall/box, capped at max_range."""
        dx, dy = math.cos(angle), math.sin(angle)
        best = max_range
        # Walls (the origin is inside the arena, so take the exit distance on each axis).
        if dx > _EPS:
            best = min(best, (self.width - ox) / dx)
        elif dx < -_EPS:
            best = min(best, -ox / dx)
        if dy > _EPS:
            best = min(best, (self.height - oy) / dy)
        elif dy < -_EPS:
            best = min(best, -oy / dy)
        for b in self.boxes:
            t = _ray_box(ox, oy, dx, dy, b)
            if t is not None and t < best:
                best = t
        return max(0.0, best)


def _ray_box(ox: float, oy: float, dx: float, dy: float, b: Box) -> Optional[float]:
    """Slab test. Returns the entry distance, 0 if the origin is inside, None on a miss."""
    tmin, tmax = -math.inf, math.inf
    for o, d, lo, hi in ((ox, dx, b.x0, b.x1), (oy, dy, b.y0, b.y1)):
        if abs(d) < _EPS:
            if o < lo or o > hi:
                return None
            continue
        t1, t2 = (lo - o) / d, (hi - o) / d
        if t1 > t2:
            t1, t2 = t2, t1
        tmin, tmax = max(tmin, t1), min(tmax, t2)
        if tmin > tmax:
            return None
    if tmax < 0:
        return None
    return max(tmin, 0.0)


def track_targets(command: str) -> tuple[float, float]:
    """Target (left, right) track speeds in m/s for a command."""
    v = config.ROVER_SPEED_MPS
    spin = math.radians(config.ROVER_TURN_DPS) * config.ROVER_TRACK_WIDTH_M / 2.0
    return {
        commands.FORWARD: (v, v),
        commands.BACK: (-v, -v),
        commands.LEFT: (-spin, spin),    # right track forward -> counter-clockwise -> left
        commands.RIGHT: (spin, -spin),
        commands.STOP: (0.0, 0.0),
    }[command]


def _ramp(current: float, target: float, dt: float) -> float:
    if current == target:
        return current
    slowing = target == 0.0 or (current != 0.0 and (current * target < 0 or abs(target) < abs(current)))
    step = (config.ROVER_BRAKE_MPS2 if slowing else config.ROVER_ACCEL_MPS2) * dt
    if target > current:
        return min(target, current + step)
    return max(target, current - step)


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


class RoverSim:
    def __init__(self, arena: Optional[Arena] = None,
                 start: tuple[float, float, float] = config.ROVER_START) -> None:
        self.arena = arena or Arena()
        self.start = start
        self.reset()

    # -- state -------------------------------------------------------------------
    def reset(self) -> None:
        x, y, heading_deg = self.start
        self.x, self.y = float(x), float(y)
        self.heading = math.radians(heading_deg)
        self.v_left = self.v_right = 0.0
        self.time = 0.0
        self.cmd = commands.STOP
        self._last_cmd_time = -math.inf
        self.link_lost = True
        self.blocked = False
        self.bumped = False
        self.trail: deque[tuple[float, float, float]] = deque([(self.x, self.y, 0.0)])
        self._sense()

    @property
    def speed(self) -> float:
        return (self.v_left + self.v_right) / 2.0

    @property
    def turn_rate(self) -> float:
        return (self.v_right - self.v_left) / config.ROVER_TRACK_WIDTH_M

    @property
    def heading_deg(self) -> float:
        return math.degrees(self.heading)

    # -- input -------------------------------------------------------------------
    def command(self, cmd: str) -> None:
        self.cmd = commands.validate(cmd)
        self._last_cmd_time = self.time
        self.link_lost = False
        self.blocked = cmd == commands.FORWARD and self.distance < config.BLOCK_DISTANCE_M

    # -- physics -----------------------------------------------------------------
    def update(self, dt: float) -> None:
        if dt <= 0:
            return
        dt = min(dt, config.SIM_MAX_FRAME_S)
        steps = max(1, math.ceil(dt / config.SIM_MAX_STEP_S))
        h = dt / steps
        for _ in range(steps):
            self._step(h)

    def _step(self, h: float) -> None:
        self.time += h
        self.link_lost = self.time - self._last_cmd_time > config.ROVER_LINK_TIMEOUT_S
        cmd = commands.STOP if self.link_lost else self.cmd
        self.blocked = cmd == commands.FORWARD and self.distance < config.BLOCK_DISTANCE_M
        if self.blocked:
            cmd = commands.STOP

        target_l, target_r = track_targets(cmd)
        self.v_left = _ramp(self.v_left, target_l, h)
        self.v_right = _ramp(self.v_right, target_r, h)

        v, w = self.speed, self.turn_rate
        mid = self.heading + w * h / 2.0
        nx = self.x + v * math.cos(mid) * h
        ny = self.y + v * math.sin(mid) * h
        self.heading = _wrap(self.heading + w * h)

        r = config.ROVER_COLLISION_RADIUS_M
        if not self.arena.circle_hits(nx, ny, r):
            self.x, self.y, self.bumped = nx, ny, False
        elif not self.arena.circle_hits(nx, self.y, r):
            self.x, self.bumped = nx, True        # slide along the obstacle
        elif not self.arena.circle_hits(self.x, ny, r):
            self.y, self.bumped = ny, True
        elif v != 0.0:
            self.bumped = True                     # stalled against something
            self.v_left = self.v_right = 0.0

        self._sense()
        self._update_trail()

    def _sense(self) -> None:
        ox = self.x + math.cos(self.heading) * config.ULTRASONIC_OFFSET_M
        oy = self.y + math.sin(self.heading) * config.ULTRASONIC_OFFSET_M
        n = max(1, config.ULTRASONIC_RAYS)
        half = math.radians(config.ULTRASONIC_BEAM_DEG) / 2.0
        angles = [self.heading] if n == 1 else [
            self.heading - half + i * (2 * half / (n - 1)) for i in range(n)]
        self.sensor_origin = (ox, oy)
        self.rays = [(a, self.arena.raycast(ox, oy, a, config.ULTRASONIC_MAX_RANGE_M)) for a in angles]
        self.distance = min(d for _, d in self.rays)

    def _update_trail(self) -> None:
        lx, ly, _ = self.trail[-1]
        if math.hypot(self.x - lx, self.y - ly) >= config.TRAIL_MIN_STEP_M:
            self.trail.append((self.x, self.y, self.time))
        while len(self.trail) > 1 and self.time - self.trail[0][2] > config.TRAIL_S:
            self.trail.popleft()


class SimSink(Sink):
    """Feeds the command stream into a RoverSim. ``update(dt)`` advances physics each frame."""

    name = "sim"

    def __init__(self, sim: Optional[RoverSim] = None) -> None:
        self.sim = sim or RoverSim()
        self.last_command: Optional[str] = None

    def send(self, command: str) -> None:
        self.last_command = command
        self.sim.command(command)

    def update(self, dt: float) -> None:
        self.sim.update(dt)

    def reset(self) -> None:
        self.sim.reset()

    @property
    def blocked(self) -> bool:
        return self.sim.blocked

    def status(self) -> SinkStatus:
        return SinkStatus(True, "SIM", "link lost" if self.sim.link_lost else "running")
