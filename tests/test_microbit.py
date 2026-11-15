"""The real-rover sink (rover/microbit.py) against a fake micro:bit: no Bluetooth, no hardware.

The link logic runs in-process on asyncio; one test runs the whole sink with its link process."""
import asyncio
import os
import signal
import sys
import time
from pathlib import Path

import pytest

import commands as C
import config
from rover.microbit import (UART_RX, UART_TX, LinkSettings, MicrobitSink, RoverLink, matches, not_found_hint,
                            short_name)

FAST = LinkSettings.from_config(keepalive_s=0.05, stale_s=0.15, write_response=False, require_ping=True,
                                ping_timeout_s=0.2, scan_s=0.05, connect_timeout_s=1.0, retry_s=0.05,
                                write_timeout_s=0.5, close_s=3.0)


# -- a fake micro:bit (stands in for bleak) ---------------------------------------------------------
class FakeDevice:
    def __init__(self, name="BBC micro:bit [tegip]", address="5A1B0C2D-0000-4000-8000-00000000CAFE"):
        self.name, self.address = name, address


class FakeChar:
    def __init__(self, uuid, properties):
        self.uuid, self.properties = uuid, list(properties)


class FakeServices:
    def __init__(self, chars):
        self.chars = {c.uuid: c for c in chars}

    def get_characteristic(self, uuid):
        return self.chars.get(uuid)


class FakeClient:
    def __init__(self, backend, device, disconnected_callback):
        self.backend, self.device, self.on_disconnect = backend, device, disconnected_callback
        self.is_connected = False
        self.notify = None
        self.services = FakeServices([FakeChar(UART_RX, backend.rx_properties), FakeChar(UART_TX, ["indicate"])])

    async def connect(self):
        await asyncio.sleep(0.001)
        self.is_connected = True

    async def disconnect(self):
        self.is_connected = False
        self.backend.disconnects += 1

    async def start_notify(self, char, callback):
        self.notify = callback

    async def write_gatt_char(self, char, data, response):
        assert self.is_connected and char.uuid == UART_RX
        await asyncio.sleep(0.002 if response else 0)
        self.backend.record(bytes(data), response)
        if data == b"P\n" and self.backend.firmware == "new" and self.notify:
            asyncio.get_running_loop().call_later(0.005, self.notify, None, bytearray(b"P\n"))

    def drop(self):
        """The link drops (out of range, micro:bit reset)."""
        self.is_connected = False
        self.on_disconnect(self)


class FakeBackend:
    """One micro:bit, BBC micro:bit [tegip]. Every write is kept (and appended to ``log`` if given)."""

    def __init__(self, firmware="new", rx_properties=("write", "write-without-response"), visible=True, log=None):
        self.firmware, self.rx_properties, self.visible, self.log = firmware, rx_properties, visible, log
        self.writes = []          # (t, line, response)
        self.clients = []
        self.disconnects = 0

    async def find(self, target, timeout):
        await asyncio.sleep(0.001)
        dev = FakeDevice()
        if self.visible and matches(target, dev.address, dev.name):
            return dev
        await asyncio.sleep(timeout)
        return None

    def client(self, device, disconnected_callback):
        c = FakeClient(self, device, disconnected_callback)
        self.clients.append(c)
        return c

    def record(self, data, response):
        line = data.decode().strip()
        self.writes.append((time.monotonic(), line, response))
        if self.log:
            with open(self.log, "a") as f:
                f.write(f"{time.monotonic():.6f} {line} {int(response)}\n")

    def lines(self):
        return [w[1] for w in self.writes]


class Harness:
    """A RoverLink on the test's event loop, with its status reports collected."""

    def __init__(self, backend, target="", settings=FAST):
        self.backend = backend
        self.states = []
        self.link = RoverLink(target, backend, lambda s, d, dev: self.states.append((s, d, dev)), settings)
        self.task = asyncio.ensure_future(self.link.run())

    @property
    def state(self):
        return self.states[-1][0] if self.states else None

    async def until(self, cond, timeout=2.0):
        t0 = time.monotonic()
        while not cond():
            assert time.monotonic() - t0 < timeout, f"timed out; states={self.states} writes={self.backend.writes}"
            await asyncio.sleep(0.005)

    async def drive(self, cmd, seconds, every=0.1):
        """The app's controller: send ``cmd`` every 0.1 s for ``seconds``."""
        t_end = time.monotonic() + seconds
        while time.monotonic() < t_end:
            self.link.command(cmd)
            await asyncio.sleep(every)

    async def close(self):
        self.link.close()
        await asyncio.wait_for(self.task, 2.0)


def run(coro):
    return asyncio.run(coro)


# -- finding the rover --------------------------------------------------------------------------------
def test_finds_the_rover_by_name_or_address_never_another_device():
    assert matches("", "x", "BBC micro:bit [tegip]")
    assert not matches("", "x", "Samarth's Phone") and not matches("", "x", None)
    assert matches("tegip", "x", "BBC micro:bit [tegip]") and matches("TEGIP", "x", "BBC micro:bit [tegip]")
    assert not matches("zuvav", "x", "BBC micro:bit [tegip]")
    assert matches("ca:8e:97:38:cf:7e", "CA:8E:97:38:CF:7E", None)                 # Linux / Windows
    assert matches("5a1b0c2d-0000-4000-8000-00000000cafe", "5A1B0C2D-0000-4000-8000-00000000CAFE", "")  # macOS
    assert short_name("BBC micro:bit [tegip]") == "tegip"


def test_a_mac_address_on_macos_explains_itself(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    assert "hides MAC" in not_found_hint("CA:8E:97:38:CF:7E")
    assert not_found_hint("tegip") == "'tegip' not found (on? in use elsewhere?)"


# -- the link -------------------------------------------------------------------------------------------
def test_writes_changes_at_once_keeps_alive_and_stops_when_the_app_goes_quiet():
    async def main():
        fb = FakeBackend()
        h = Harness(fb)
        await h.until(lambda: h.state == "ready")
        assert h.states[-1][2] == "tegip" and "ping" in h.states[-1][1]
        assert fb.lines()[0] == "P"                                     # ping first, nothing else until it answers

        h.link.command("F")
        t_sent = time.monotonic()
        await h.until(lambda: "F" in fb.lines())
        t_first = next(t for t, line, _ in fb.writes if line == "F")
        assert t_first - t_sent < 0.02                                  # a change goes out at once
        await h.drive("F", 0.5)
        f_times = [t for t, line, _ in fb.writes if line == "F"]
        gaps = [b - a for a, b in zip(f_times, f_times[1:])]
        assert max(gaps) < 0.09                                         # keepalive (0.05 s) well inside the watchdog
        assert all(not resp for _, line, resp in fb.writes if line == "F")   # without response when it can

        n = len(fb.writes)                                              # the app goes quiet (UI stalled)
        await asyncio.sleep(0.3)
        after = fb.lines()[n:]
        assert "S" in after and after[-1] == "S"
        t_stop = next(t for t, line, _ in fb.writes[n:] if line == "S")
        assert t_stop - f_times[-1] < FAST.stale_s + 0.06                 # stopped by the link itself

        h.link.command("L")                                             # the app is back
        await h.until(lambda: fb.lines()[-1] == "L")
        await h.close()
        assert fb.writes[-1][1:] == ("S", True)                         # final S, acknowledged
        assert fb.disconnects == 1 and h.state == "closed"
    run(main())


def test_old_firmware_gets_one_stop_and_never_a_drive_command():
    async def main():
        fb = FakeBackend(firmware="old")
        h = Harness(fb)
        await h.until(lambda: h.state == "old-firmware")
        assert "flash firmware/maqueen.ts" in h.states[-1][1]
        await h.drive("F", 0.4)
        await h.close()
        assert set(fb.lines()) == {"P", "S"}
        assert fb.lines().count("S") == 2                               # one hold, one final
    run(main())


def test_reconnects_after_the_link_drops():
    async def main():
        fb = FakeBackend()
        h = Harness(fb)
        await h.until(lambda: h.state == "ready")
        h.link.command("F")
        await h.until(lambda: "F" in fb.lines())
        fb.clients[0].drop()
        await h.until(lambda: len(fb.clients) == 2 and h.state == "ready")
        assert any(s == "scanning" and "lost" in d for s, d, _ in h.states)
        h.link.command("R")
        await h.until(lambda: fb.lines()[-1] == "R")
        await h.close()
    run(main())


def test_writes_with_response_when_that_is_all_the_micro_bit_offers():
    async def main():
        fb = FakeBackend(rx_properties=("write",))
        h = Harness(fb)
        await h.until(lambda: h.state == "ready")
        await h.drive("F", 0.2)
        await h.close()
        assert all(resp for _, _, resp in fb.writes)
    run(main())


def test_keeps_scanning_and_closes_promptly_when_nothing_is_there():
    async def main():
        fb = FakeBackend(visible=False)
        h = Harness(fb, target="zuvav", settings=LinkSettings.from_config(**{**FAST.__dict__, "scan_s": 5.0}))
        await asyncio.sleep(0.05)
        assert h.state == "scanning"
        t0 = time.monotonic()
        await h.close()                                                 # mid-scan: no waiting for the scan to end
        assert time.monotonic() - t0 < 0.5
        assert fb.writes == [] and fb.clients == []
    run(main())


# -- the sink, with its link process --------------------------------------------------------------------
def _log_lines(path):
    if not path.exists():
        return []
    return [line.split() for line in path.read_text().splitlines()]


def _wait(cond, timeout=10.0):
    t0 = time.monotonic()
    while not cond():
        assert time.monotonic() - t0 < timeout
        time.sleep(0.02)


def test_sink_drives_through_its_link_process_and_stops_on_close(tmp_path):
    log = tmp_path / "writes.txt"
    sink = MicrobitSink("tegip", backend=FakeBackend(log=str(log)), settings=FAST)
    try:
        assert not sink.status().ok and sink.status().label == "ROVER"
        _wait(lambda: sink.status().ok)                                 # spawned, found, connected, pinged
        st = sink.status()
        assert st.label == "ROVER tegip" and "ping" in st.detail

        t0 = time.perf_counter()
        for _ in range(10):
            sink.send("F")
            time.sleep(0.05)
        assert (time.perf_counter() - t0) < 0.5 + 0.2                   # sends didn't wait
        _wait(lambda: any(w[1] == "F" for w in _log_lines(log)))
        sink.send("S")
        _wait(lambda: _log_lines(log)[-1][1] == "S")
    finally:
        sink.close()
    lines = _log_lines(log)
    assert lines[-1][1:] == ["S", "1"]                                  # final S, acknowledged
    assert {w[1] for w in lines} <= {"P", f"T{config.MICROBIT_TURN_SPEED}", "F", "S"}
    assert [w[1] for w in lines][:2] == ["P", f"T{config.MICROBIT_TURN_SPEED}"]   # turn speed once, after the ping
    assert not sink.status().ok


def test_send_never_blocks_even_if_the_link_process_is_stuck(tmp_path):
    sink = MicrobitSink("tegip", backend=FakeBackend(), settings=FAST)
    pid = sink._proc.pid
    try:
        _wait(lambda: sink.state == "ready")
        os.kill(pid, signal.SIGSTOP)                                    # stuck: not reading its pipe
        t0 = time.perf_counter()
        worst = 0.0
        for _ in range(30000):                                          # far more than the pipe holds
            t = time.perf_counter()
            sink.send(C.FORWARD)
            worst = max(worst, time.perf_counter() - t)
        assert sink.dropped > 0                                         # the pipe filled up...
        assert worst < 0.05 and time.perf_counter() - t0 < 5.0          # ...and send() still never waited
    finally:
        os.kill(pid, signal.SIGKILL)
        sink.close()


def test_main_has_a_rover_flag():
    import main
    assert main.parse_args([]).rover is None
    assert main.parse_args(["--rover"]).rover == ""
    assert main.parse_args(["--rover", "tegip", "--source", "keyboard"]).rover == "tegip"
