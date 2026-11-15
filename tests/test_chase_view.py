"""Chase-cam renderer: projection math, near-plane clipping, painter's order and headless draws."""
import math

import numpy as np
import pygame
import pytest

import commands as C
import config
from sinks.sim_sink import RoverSim
from ui import chase_view as cv
from ui import theme
from ui.chase_view import ChaseCamera, ChaseView
from ui.theme import Fonts

VIEW = pygame.Rect(24, 98, 912, 778)        # the 3D viewport ChaseView uses at 1600x900
SENTINEL = (1, 2, 3)


@pytest.fixture(scope="module", autouse=True)
def _pygame_fonts():
    pygame.font.init()
    yield


@pytest.fixture
def fonts():
    return Fonts()


def make_sim(x, y, deg, cmd=None, dt=0.0):
    sim = RoverSim(start=(x, y, deg))
    if cmd:
        sim.command(cmd)
    if dt:
        sim.update(dt)
    return sim


def drive(sim, plan, dt=1 / 60):
    for cmd, secs in plan:
        for _ in range(int(secs / dt)):
            sim.command(cmd)
            sim.update(dt)


def ahead(x, y, yaw, d, side=0.0):
    """World point ``d`` ahead and ``side`` to the left of a pose."""
    return (x + math.cos(yaw) * d - math.sin(yaw) * side, y + math.sin(yaw) * d + math.cos(yaw) * side)


# -- projection math ------------------------------------------------------------------------------
@pytest.mark.parametrize("deg", [0, 37, 90, 180, 270, -135])
def test_ahead_is_up_and_left_is_left(deg):
    yaw = math.radians(deg)
    cam = ChaseCamera(2.0, 1.5, yaw, VIEW)
    rx, ry = cam.project(2.0, 1.5)
    fx, fy = cam.project(*ahead(2.0, 1.5, yaw, 0.5))
    lx, ly = cam.project(*ahead(2.0, 1.5, yaw, 0.0, side=0.3))
    gx, gy = cam.project(*ahead(2.0, 1.5, yaw, 0.0, side=-0.3))
    assert fy < ry - 20 and abs(fx - rx) < 1e-6          # straight ahead -> straight up the screen
    assert lx < rx - 20 and gx > rx + 20                  # rover's left/right -> screen left/right
    assert abs((rx - lx) - (gx - rx)) < 1e-6              # symmetric about the rover


@pytest.mark.parametrize("deg", [0, 90, 225])
def test_farther_points_climb_towards_the_top_with_perspective(deg):
    yaw = math.radians(deg)
    cam = ChaseCamera(1.0, 1.0, yaw, VIEW)
    ys = [cam.project(*ahead(1.0, 1.0, yaw, d))[1] for d in (0.0, 0.5, 1.0, 1.5, 2.0)]
    steps = np.diff(ys)
    assert np.all(steps < 0)                              # monotonically up the screen
    assert np.all(np.diff(-steps) < 0)                    # foreshortening: equal metres, fewer pixels
    # A 1 m wide bar looks narrower far away than at the rover.
    w_near = cam.project(*ahead(1.0, 1.0, yaw, 0, -0.5))[0] - cam.project(*ahead(1.0, 1.0, yaw, 0, 0.5))[0]
    w_far = cam.project(*ahead(1.0, 1.0, yaw, 2, -0.5))[0] - cam.project(*ahead(1.0, 1.0, yaw, 2, 0.5))[0]
    assert w_near > w_far > 0.5 * w_near                  # perspective, but mild


def test_rover_sits_low_and_centred_and_heights_rise():
    cam = ChaseCamera(1.0, 1.0, 0.3, VIEW)
    x, y = cam.project(1.0, 1.0)
    assert abs(x - VIEW.centerx) < 1e-6
    assert VIEW.top + 0.6 * VIEW.h < y < VIEW.bottom - 0.1 * VIEW.h
    assert cam.project(1.0, 1.0, 0.2)[1] < y               # +z goes up the screen
    assert cam.pz > 1.0 and cam.scale_at(1.0, 1.0) > 0      # camera is above, rover is in front


def test_compass_direction_matches_projection():
    """The compass needle uses (-cos yaw, -sin yaw) for world +y; check it against the camera."""
    for deg in (0, 60, 135, 250):
        yaw = math.radians(deg)
        cam = ChaseCamera(2.0, 1.5, yaw, VIEW)
        a, b = cam.project(2.0, 1.5), cam.project(2.0, 1.6)
        v = np.subtract(b, a)
        v /= np.linalg.norm(v)
        assert np.allclose(v, (-math.cos(yaw), -math.sin(yaw)), atol=0.25)


# -- near-plane clipping ------------------------------------------------------------------------------
def test_points_behind_the_camera_are_rejected_not_inverted():
    cam = ChaseCamera(2.0, 1.5, 0.0, VIEW, pitch_deg=20.0)
    behind = (cam.px - 5.0, cam.py, 0.0)                 # well behind (camera faces +x)
    assert cam.to_cam(*behind)[2] < 0
    assert cam.project(*behind) is None
    assert cam.scale_at(*behind) == 0.0
    assert cam.segment(behind, (cam.px - 6.0, cam.py, 0.0)) is None
    assert cam.polygon([behind, (cam.px - 6, cam.py, 0), (cam.px - 6, cam.py + 1, 0)]) == []


def test_polygon_straddling_the_near_plane_is_clipped_to_finite_points():
    cam = ChaseCamera(2.0, 1.5, 0.0, VIEW, pitch_deg=20.0)
    quad = [(cam.px - 4, cam.py - 1, 0), (cam.px + 4, cam.py - 1, 0),
            (cam.px + 4, cam.py + 1, 0), (cam.px - 4, cam.py + 1, 0)]
    pts = cam.polygon(quad)
    assert len(pts) >= 3
    assert all(math.isfinite(c) and abs(c) < 1e7 for p in pts for c in p)
    # Visible part projects below the horizon (the floor never flips above it).
    horizon = VIEW.centery - cam.focal * math.tan(math.radians(20.0))
    assert all(p[1] > horizon for p in pts)
    seg = cam.segment((cam.px - 4, cam.py, 0), (cam.px + 4, cam.py, 0))
    assert seg is not None and all(math.isfinite(c) for p in seg for c in p)
    # The clipped end lies on the near plane, the other end is unchanged.
    assert seg[1] == pytest.approx(cam.project(cam.px + 4, cam.py, 0))


# -- painter's order -----------------------------------------------------------------------------------
def test_walls_drawn_behind_boxes_unless_the_camera_is_outside_that_wall():
    view = ChaseView(Fonts())
    sim = RoverSim()
    prisms = view._prisms(sim.arena)
    south, north, west, east = prisms[:4]
    boxes = prisms[4:]
    inside = cv.sort_far_to_near(prisms, 2.0, 1.5)
    assert all(inside.index(w) < inside.index(b) for w in (south, north, west, east) for b in boxes)
    outside = cv.sort_far_to_near(prisms, 2.0, -0.4)       # camera south of the arena
    assert all(outside.index(south) > outside.index(b) for b in boxes)
    assert all(outside.index(north) < outside.index(b) for b in boxes)


def test_void_regions_cover_only_what_the_floor_does_not():
    rect = pygame.Rect(0, 0, 100, 100)
    assert cv._outside_regions([(-10, -10), (110, -10), (110, 110), (-10, 110)], rect) == []
    regions = cv._outside_regions([(50, -10), (110, -10), (110, 110), (50, 110)], rect)  # right half is floor
    assert regions
    for poly in regions:
        assert max(x for x, _ in poly) <= 50 + 1e-9


# -- camera follow ---------------------------------------------------------------------------------------
def test_camera_lags_gently_while_spinning_then_settles(fonts):
    sim = make_sim(2.0, 1.0, 90)
    view = ChaseView(fonts)
    surf = pygame.Surface((960, 900))
    for _ in range(30):
        sim.command(C.LEFT)
        sim.update(1 / 60)
        view.draw(surf, surf.get_rect(), sim)
    lag = math.degrees(cv._wrap(sim.heading - view.camera.yaw))
    assert 0 < lag < 25                                    # follows a left spin, slightly behind
    for _ in range(90):
        sim.command(C.STOP)
        sim.update(1 / 60)
        view.draw(surf, surf.get_rect(), sim)
    assert view.camera.yaw == pytest.approx(sim.heading, abs=1e-3)


def test_camera_snaps_after_reset(fonts):
    sim = RoverSim()
    view = ChaseView(fonts)
    surf = pygame.Surface((960, 900))
    drive(sim, [(C.FORWARD, 1.0), (C.LEFT, 1.0), (C.FORWARD, 1.0)])
    view.draw(surf, surf.get_rect(), sim)
    sim.reset()                                            # F5: time goes back to 0, pose jumps
    view.draw(surf, surf.get_rect(), sim)
    _, _, x, y, yaw = view._cam_state
    assert (x, y, yaw) == pytest.approx((sim.x, sim.y, sim.heading))
    # Same for a teleport without a time reset.
    sim.x, sim.y, sim.heading = 3.5, 2.5, math.radians(200)
    sim.update(1 / 60)
    view.draw(surf, surf.get_rect(), sim)
    assert view.camera.yaw == pytest.approx(cv._wrap(sim.heading))


# -- headless draws ----------------------------------------------------------------------------------------
def _draw(fonts, sim, size=(960, 900), banners=(), offset=(20, 10)):
    """Draw into a rect inside a sentinel-filled surface; return (surface, rect, view)."""
    surf = pygame.Surface((size[0] + 2 * offset[0], size[1] + 2 * offset[1]))
    surf.fill(SENTINEL)
    rect = pygame.Rect(offset, size)
    view = ChaseView(fonts)
    view.draw(surf, rect, sim, banners)
    return surf, rect, view


def _assert_inside_rect_only(surf, rect):
    px = pygame.surfarray.array3d(surf)
    outside = np.ones(px.shape[:2], bool)
    outside[rect.left:rect.right, rect.top:rect.bottom] = False
    assert np.all(px[outside] == SENTINEL), "drew outside its rect"


def _has_colour(surf, rect, colour) -> bool:
    px = pygame.surfarray.array3d(surf.subsurface(rect))
    return bool(np.any(np.all(px == colour, axis=-1)))


POSES = {
    "centre_east": (2.0, 1.0, 0),
    "facing_north": (2.0, 0.8, 90),
    "facing_west": (3.0, 2.2, 180),
    "facing_south": (2.0, 2.4, 270),
    "diagonal": (1.9, 1.0, 33),
    "corner_facing_in": (0.12, 0.12, 225),
    "corner_back_in": (0.1, 0.1, 45),             # camera ends up outside the arena
    "far_corner_back_in": (3.9, 2.9, 225),
    "wall_at_back": (3.9, 1.5, 180),
}


@pytest.mark.parametrize("name", POSES)
def test_smoke_draw_poses(fonts, name):
    x, y, deg = POSES[name]
    sim = make_sim(x, y, deg, C.LEFT if deg % 90 else C.FORWARD, 0.05)
    surf, rect, view = _draw(fonts, sim)
    _assert_inside_rect_only(surf, rect)
    cam = view.camera
    assert cam is not None
    # The rover is always on screen, inside the 3D viewport, and drawn on top of everything.
    rx, ry = cam.project(sim.x, sim.y, config.ROVER_LENGTH_M)
    assert rect.collidepoint(rx, ry)
    assert _has_colour(surf, pygame.Rect(round(rx) - 60, round(ry) - 60, 120, 120), theme.ROVER_BODY)
    assert _has_colour(surf, rect, theme.FLOOR)


def test_corner_puts_camera_outside_arena_and_still_draws(fonts):
    sim = make_sim(0.1, 0.1, 45)
    surf, rect, view = _draw(fonts, sim)
    assert view.camera.px < 0 and view.camera.py < 0
    _assert_inside_rect_only(surf, rect)


def test_low_pitch_puts_walls_behind_the_camera_without_crashing(fonts, monkeypatch):
    monkeypatch.setattr(cv, "CHASE_PITCH_DEG", 22.0)
    sim = RoverSim()
    drive(sim, [(C.FORWARD, 2.0), (C.RIGHT, 0.5), (C.FORWARD, 2.0)])
    for x, y, deg in [(3.8, 1.5, 0), (2.0, 2.8, 90), (0.2, 0.2, 225), (3.9, 2.9, 45)]:
        sim.x, sim.y, sim.heading = x, y, math.radians(deg)
        sim._sense()
        surf, rect, view = _draw(fonts, sim)
        cam = view.camera
        assert cam.to_cam(*_far_corner(sim, cam), 0.0)[2] < cam.near   # something really is behind
        _assert_inside_rect_only(surf, rect)


def _far_corner(sim, cam):
    corners = [(0, 0), (sim.arena.width, 0), (0, sim.arena.height), (sim.arena.width, sim.arena.height)]
    return min(corners, key=lambda c: cam.to_cam(c[0], c[1], 0.0)[2])


def test_blocked_against_box_shows_banner_and_red_sonar(fonts):
    sim = RoverSim()                                     # start pose faces a box
    drive(sim, [(C.FORWARD, 6.0)])
    assert sim.blocked
    surf, rect, view = _draw(fonts, sim)
    _assert_inside_rect_only(surf, rect)
    assert _has_colour(surf, rect, theme.BAD)
    hit = view.camera.project(sim.x + sim.distance + config.ULTRASONIC_OFFSET_M, sim.y)
    assert rect.collidepoint(hit)


def test_no_signal_and_passed_in_banners(fonts):
    sim = make_sim(2.0, 1.0, 90, dt=0.6)                # never commanded -> link lost
    assert sim.link_lost
    surf, rect, _ = _draw(fonts, sim, banners=[("MOCK: something", (0, 200, 200))])
    assert _has_colour(surf, rect, theme.WARN)
    assert _has_colour(surf, rect, (0, 200, 200))


def test_sim_none_shows_simulation_off(fonts):
    surf, rect, view = _draw(fonts, None, banners=[("EEG: no headset", theme.BAD)])
    _assert_inside_rect_only(surf, rect)
    assert view.camera is None
    assert _has_colour(surf, rect, theme.BAD)
    assert _has_colour(surf, rect, theme.PANEL)


@pytest.mark.parametrize("size", [(500, 430), (360, 300), (120, 100)])
def test_small_rects(fonts, size):
    sim = RoverSim()
    drive(sim, [(C.FORWARD, 1.0), (C.LEFT, 0.5), (C.FORWARD, 1.0)])
    surf, rect, view = _draw(fonts, sim, size=size)
    _assert_inside_rect_only(surf, rect)


def test_long_trail_draws(fonts):
    sim = RoverSim()
    drive(sim, [(C.FORWARD, 3), (C.LEFT, 1), (C.FORWARD, 3), (C.LEFT, 1), (C.FORWARD, 3), (C.RIGHT, 2),
                (C.FORWARD, 4)])
    assert len(sim.trail) > 200
    surf, rect, _ = _draw(fonts, sim)
    _assert_inside_rect_only(surf, rect)
    assert _has_colour(surf, rect, theme.lerp_color(theme.TRAIL, theme.FLOOR, (0.5 / cv.CHASE_TRAIL_BANDS) ** 0.8))
