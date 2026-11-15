import math

import pytest

import commands as C
import config
from sinks.sim_sink import Arena, RoverSim

FRAME = 1 / 60


def drive(sim: RoverSim, cmd: str, seconds: float, hz: float = config.SEND_HZ) -> None:
    """Run the sim for ``seconds`` while (re)sending ``cmd`` at ``hz``, like the controller does."""
    t, next_send = 0.0, 0.0
    while t < seconds - 1e-9:
        if t >= next_send - 1e-9:
            sim.command(cmd)
            next_send += 1.0 / hz
        sim.update(FRAME)
        t += FRAME


def open_sim(x=2.0, y=1.5, heading=0.0) -> RoverSim:
    return RoverSim(Arena(boxes=()), start=(x, y, heading))


def test_forward_moves_along_heading():
    for heading in (0.0, 30.0, 135.0, -60.0):
        sim = open_sim(heading=heading)
        drive(sim, C.FORWARD, 1.0)
        dx, dy = sim.x - 2.0, sim.y - 1.5
        assert math.hypot(dx, dy) > 0.2
        assert math.degrees(math.atan2(dy, dx)) == pytest.approx(heading, abs=0.01)
        assert sim.heading_deg == pytest.approx(heading, abs=0.01)


def test_forward_reaches_configured_speed_after_ramp():
    sim = open_sim()
    drive(sim, C.FORWARD, 0.1)
    assert 0 < sim.speed < config.ROVER_SPEED_MPS  # still ramping up
    drive(sim, C.FORWARD, 1.0)
    assert sim.speed == pytest.approx(config.ROVER_SPEED_MPS)


def test_back_moves_opposite_to_heading():
    sim = open_sim(heading=90.0)
    drive(sim, C.BACK, 1.0)
    assert sim.y < 1.5 - 0.2
    assert sim.x == pytest.approx(2.0, abs=1e-9)


def test_left_turns_rover_to_its_left_in_place():
    sim = open_sim(heading=0.0)  # facing +x; its left is +y
    drive(sim, C.LEFT, 1.0)
    assert 80.0 < sim.heading_deg < 92.0  # ~90 deg/s minus the short ramp
    assert (sim.x, sim.y) == pytest.approx((2.0, 1.5), abs=1e-9)  # spin in place
    drive(sim, C.FORWARD, 1.0)
    # Now driving towards the rover's original left (+y), within ~15 deg.
    assert sim.y > 1.5 + 0.2
    assert sim.y - 1.5 > 3.5 * abs(sim.x - 2.0)


def test_right_turns_rover_to_its_right():
    sim = open_sim(heading=0.0)
    drive(sim, C.RIGHT, 1.0)
    assert -92.0 < sim.heading_deg < -80.0
    assert sim.turn_rate < 0


def test_stop_brakes_to_zero_and_stays_put():
    sim = open_sim()
    drive(sim, C.FORWARD, 1.0)
    drive(sim, C.STOP, 0.5)
    assert sim.v_left == 0.0 and sim.v_right == 0.0
    x, y = sim.x, sim.y
    drive(sim, C.STOP, 1.0)
    assert (sim.x, sim.y) == (x, y)


def test_link_watchdog_stops_rover_when_commands_stop_arriving():
    sim = open_sim()
    sim.command(C.FORWARD)  # sent once, never again
    for _ in range(int(2.0 / FRAME)):
        sim.update(FRAME)
    assert sim.link_lost
    assert sim.speed == 0.0
    assert sim.x - 2.0 < config.ROVER_SPEED_MPS * (config.ROVER_LINK_TIMEOUT_S + 0.2)


def test_ultrasonic_measures_distance_to_walls_and_boxes():
    sim = open_sim(heading=0.0)
    assert sim.distance == pytest.approx(config.ARENA_W_M - 2.0 - config.ULTRASONIC_OFFSET_M)
    sim = RoverSim(Arena(boxes=[(3.0, 1.0, 3.5, 2.0)]), start=(1.0, 1.5, 0.0))
    assert sim.distance == pytest.approx(3.0 - 1.0 - config.ULTRASONIC_OFFSET_M)
    sim = open_sim(heading=90.0)
    assert sim.distance == pytest.approx(config.ARENA_H_M - 1.5 - config.ULTRASONIC_OFFSET_M)


def test_forward_refused_when_obstacle_closer_than_threshold():
    front = 2.0 + config.ULTRASONIC_OFFSET_M
    sim = RoverSim(Arena(boxes=[(front + 0.10, 1.0, front + 0.5, 2.0)]), start=(2.0, 1.5, 0.0))
    drive(sim, C.FORWARD, 1.0)
    assert sim.blocked
    assert sim.x == pytest.approx(2.0)
    drive(sim, C.BACK, 0.5)  # other commands still work
    assert not sim.blocked
    assert sim.x < 2.0 - 0.05


def test_driving_at_obstacle_stops_short_of_it():
    sim = RoverSim(Arena(boxes=[(3.0, 1.0, 3.5, 2.0)]), start=(1.0, 1.5, 0.0))
    drive(sim, C.FORWARD, 15.0)
    assert sim.blocked
    assert 0.08 < sim.distance < config.BLOCK_DISTANCE_M
    assert sim.speed == 0.0
    assert not sim.bumped


def test_backing_into_wall_is_stopped_by_collision():
    sim = open_sim(x=0.5, heading=0.0)
    drive(sim, C.BACK, 5.0)
    assert sim.x >= config.ROVER_COLLISION_RADIUS_M - 1e-9
    assert sim.bumped


def test_reset_restores_start_pose():
    sim = open_sim(heading=45.0)
    drive(sim, C.FORWARD, 1.0)
    sim.reset()
    assert (sim.x, sim.y, sim.heading_deg) == pytest.approx((2.0, 1.5, 45.0))
    assert sim.speed == 0.0 and len(sim.trail) == 1


def test_rejects_non_contract_commands():
    with pytest.raises(ValueError):
        open_sim().command("X")
