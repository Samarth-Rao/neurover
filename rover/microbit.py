"""The real rover: a BBC micro:bit on a DFRobot Maqueen, over Bluetooth LE UART (firmware/maqueen.ts).

    python main.py --rover            # the first "BBC micro:bit [.....]" seen
    python main.py --rover tegip      # that one (part of its name, or its address)

The Bluetooth link runs in its own process (``rover-link``), started by MicrobitSink:

    UI thread  --send("F")-->  pipe (non-blocking)  -->  rover-link process: asyncio + bleak
                                                              writes "F\\n" to the micro:bit's UART

Why a process and not a thread: inside the app's process the link has to share Python's GIL with
the UI and the BrainBit callbacks, and handing a command to a thread then took 6 ms typical and
up to ~50 ms when the UI was busy; to a process it takes ~0.1 ms whatever the UI does. It also keeps
bleak's CoreBluetooth stack out of the process that runs the BrainBit SDK.

What goes over the air (a line per write, ``commands.encode``):
  * the current command at once when it changes, and again every MICROBIT_KEEPALIVE_S (well inside
    the firmware's 500 ms watchdog);
  * S as soon as the app goes quiet for MICROBIT_STALE_S (UI stalled or crashed): the link never
    keeps the rover driving on its own;
  * on close (or if the app dies), a final S, acknowledged, then a disconnect.
After connecting it pings ("P"; the firmware answers "P"). Firmware that doesn't answer is the old one,
which needs 0.6 s per line and falls seconds behind a 10 Hz stream, so with MICROBIT_REQUIRE_PING it
only ever gets one S.
"""
from __future__ import annotations

import asyncio
import math
import multiprocessing as mp
import os
import re
import signal
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import commands
import config
from .sim import Sink, SinkStatus

# The micro:bit's UART service. Its naming is from the micro:bit's side: we write its RX.
UART_SERVICE = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
UART_RX = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"   # laptop -> micro:bit (write / write without response)
UART_TX = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"   # micro:bit -> laptop (indicate)
PING = b"P\n"
NAME_PREFIX = "BBC micro:bit"

READY = "ready"   # connected and the firmware answered: driving
_MAC = re.compile(r"^([0-9a-f]{2}:){5}[0-9a-f]{2}$", re.I)
_MAX_DETAIL = 160  # keeps every status message one atomic pipe write


def matches(target: str, address: str, name: Optional[str]) -> bool:
    """Is this advertiser the rover? "" = any micro:bit; else its address or part of its name."""
    name = name or ""
    t = (target or "").strip().lower()
    if not t:
        return name.lower().startswith(NAME_PREFIX.lower())
    return (address or "").lower() == t or (bool(name) and t in name.lower())


def short_name(name: Optional[str]) -> str:
    """"BBC micro:bit [tegip]" -> "tegip"."""
    m = re.search(r"\[(\w+)\]", name or "")
    return m.group(1) if m else (name or "")


def not_found_hint(target: str) -> str:
    """Short enough for the dashboard's status line."""
    if _MAC.match(target or "") and sys.platform == "darwin":
        return "macOS hides MAC addresses: --rover NAME instead"   # tools/microbit_latency.py --scan lists names
    return f"'{target}' not found (on? in use elsewhere?)" if target else "no micro:bit found (on? in use elsewhere?)"


@dataclass(frozen=True)
class LinkSettings:
    keepalive_s: float
    stale_s: float
    write_response: bool
    require_ping: bool
    ping_timeout_s: float
    scan_s: float
    connect_timeout_s: float
    retry_s: float
    write_timeout_s: float
    close_s: float
    turn_speed: int = 0          # sent once on connect as "T<speed>" (firmware/maqueen.ts); 0 = leave the firmware's

    @classmethod
    def from_config(cls, **overrides: Any) -> "LinkSettings":
        values = dict(
            keepalive_s=config.MICROBIT_KEEPALIVE_S, stale_s=config.MICROBIT_STALE_S,
            write_response=config.MICROBIT_WRITE_RESPONSE, require_ping=config.MICROBIT_REQUIRE_PING,
            ping_timeout_s=config.MICROBIT_PING_TIMEOUT_S, scan_s=config.MICROBIT_SCAN_S,
            connect_timeout_s=config.MICROBIT_CONNECT_TIMEOUT_S, retry_s=config.MICROBIT_RETRY_S,
            write_timeout_s=config.MICROBIT_WRITE_TIMEOUT_S, close_s=config.MICROBIT_CLOSE_S,
            turn_speed=int(getattr(config, "MICROBIT_TURN_SPEED", 0) or 0))
        values.update(overrides)
        return cls(**values)


class BleakBackend:
    """Finds and connects with bleak (imported in the link process only)."""

    def __init__(self, connect_timeout_s: float = 10.0) -> None:
        self.connect_timeout_s = connect_timeout_s

    async def find(self, target: str, timeout: float):
        from bleak import BleakScanner
        return await BleakScanner.find_device_by_filter(
            lambda d, ad: matches(target, d.address, ad.local_name or d.name), timeout=timeout)

    def client(self, device, disconnected_callback: Callable[[Any], None]):
        from bleak import BleakClient
        return BleakClient(device, disconnected_callback=disconnected_callback, timeout=self.connect_timeout_s)


class LinkError(RuntimeError):
    pass


class RoverLink:
    """The link process's side: scan, connect, write, reconnect. Runs on one asyncio loop.

    ``command(cmd)`` is called for every command the app sends (each one also says "the app is
    alive"); ``close()`` asks for a final S and a clean exit. ``report(state, detail, device)``
    is called on every change of state.
    """

    def __init__(self, target: str, backend, report: Callable[[str, str, str], None],
                 settings: LinkSettings, clock: Callable[[], float] = time.monotonic) -> None:
        self.target = target or ""
        self.backend = backend
        self._report_cb = report
        self.s = settings
        self.clock = clock
        self.device = ""                 # short name of the micro:bit, once found
        self.writes: int = 0
        self.ping_s: Optional[float] = None
        self._cmd = commands.STOP
        self._fresh_at = -math.inf
        self._wake = asyncio.Event()
        self._closed = asyncio.Event()
        self._closing = False
        self._driving = False
        self._session: Optional[asyncio.Task] = None
        self._last_report: Optional[tuple[str, str, str]] = None
        self._misses = 0                 # scans in a row that found nothing

    # -- from the app ---------------------------------------------------------------------------
    def command(self, cmd: str) -> None:
        if not commands.is_command(cmd):
            return
        self._cmd = cmd
        self._fresh_at = self.clock()
        self._wake.set()

    def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        self._closed.set()
        self._wake.set()
        if self._session is not None and not self._driving:
            self._session.cancel()       # scanning / connecting / pinging: stop now (a final S still goes out)

    def effective(self, now: Optional[float] = None) -> str:
        """What the rover should be doing: the app's command while it keeps talking, else S."""
        now = self.clock() if now is None else now
        return self._cmd if now - self._fresh_at < self.s.stale_s else commands.STOP

    # -- main loop ------------------------------------------------------------------------------
    async def run(self) -> None:
        while not self._closing:
            self._session = asyncio.ensure_future(self._session_once())
            await asyncio.wait({self._session})
            if not self._session.cancelled() and self._session.exception() is not None:
                exc = self._session.exception()
                self._report("error", f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__)
            if self._closing:
                break
            try:
                await asyncio.wait_for(self._closed.wait(), self.s.retry_s)
            except asyncio.TimeoutError:
                pass
        self._report("closed", "closed")

    async def _session_once(self) -> None:
        self._report("scanning", not_found_hint(self.target) if self._misses else
                     f"looking for {repr(self.target) if self.target else 'a micro:bit'}")
        device = await self.backend.find(self.target, self.s.scan_s)
        if device is None:
            self._misses += 1
            self._report("scanning", not_found_hint(self.target))
            return
        self._misses = 0
        name = getattr(device, "name", None) or getattr(device, "address", "?")
        self.device = short_name(name)
        self._report("connecting", f"connecting to {name}")
        client = self.backend.client(device, self._on_disconnect)
        rx = None
        try:
            await asyncio.wait_for(client.connect(), self.s.connect_timeout_s)
            rx = client.services.get_characteristic(UART_RX)
            if rx is None:
                raise LinkError("no UART service on it: flash firmware/maqueen.ts")
            response = self.s.write_response or "write-without-response" not in rx.properties
            self._report("connecting", "connected, pinging the firmware")
            self.ping_s = await self._ping(client, rx)
            firmware_ok = self.ping_s is not None or not self.s.require_ping
            if self.ping_s is not None and self.s.turn_speed:      # turn speed: only firmware that answers pings knows "T"
                await asyncio.wait_for(client.write_gatt_char(rx, f"T{max(0, min(255, self.s.turn_speed))}\n".encode(),
                                                              response=response), self.s.write_timeout_s)
            if self.ping_s is not None:
                self._report(READY, f"ping {self.ping_s * 1000:.0f} ms")
            elif firmware_ok:
                self._report(READY, "connected (no ping reply)")
            else:
                self._report("old-firmware", "old firmware? flash firmware/maqueen.ts")   # holds S
            await self._drive(client, rx, response, firmware_ok)
            if not self._closing:
                self._report("scanning", "connection lost; reconnecting")
        finally:
            self._driving = False
            if client.is_connected:
                if rx is not None:
                    await self._final_stop(client, rx)
                try:
                    await asyncio.wait_for(client.disconnect(), 2.0)
                except Exception:
                    pass

    async def _ping(self, client, rx) -> Optional[float]:
        tx = client.services.get_characteristic(UART_TX)
        if tx is None:
            return None
        got = asyncio.Event()
        buf = bytearray()

        def on_data(_char, data: bytearray) -> None:
            buf.extend(data)
            if PING in buf:
                got.set()

        try:
            await client.start_notify(tx, on_data)
        except Exception:
            return None
        t0 = time.perf_counter()
        await asyncio.wait_for(client.write_gatt_char(rx, PING, response="write-without-response" not in rx.properties),
                               self.s.write_timeout_s)
        try:
            await asyncio.wait_for(got.wait(), self.s.ping_timeout_s)
        except asyncio.TimeoutError:
            return None
        return time.perf_counter() - t0

    async def _drive(self, client, rx, response: bool, firmware_ok: bool) -> None:
        last: Optional[str] = None
        last_t = -math.inf
        self._driving = True
        while client.is_connected and not self._closing:
            self._wake.clear()
            now = self.clock()
            cmd = self.effective(now) if firmware_ok else commands.STOP
            if cmd != last or (firmware_ok and now - last_t >= self.s.keepalive_s):
                await asyncio.wait_for(client.write_gatt_char(rx, commands.encode(cmd), response=response),
                                       self.s.write_timeout_s)
                self.writes += 1
                last, last_t = cmd, now
            if not firmware_ok:
                await self._wake.wait()
                continue
            deadline = last_t + self.s.keepalive_s
            if cmd != commands.STOP:
                deadline = min(deadline, self._fresh_at + self.s.stale_s)   # wake up to stop if the app goes quiet
            try:
                await asyncio.wait_for(self._wake.wait(), max(0.0, deadline - self.clock()))
            except asyncio.TimeoutError:
                pass

    async def _final_stop(self, client, rx) -> None:
        ack = "write" in rx.properties
        try:
            await asyncio.wait_for(client.write_gatt_char(rx, commands.encode(commands.STOP), response=ack),
                                   self.s.write_timeout_s)
            self.writes += 1
            if not ack:
                await asyncio.sleep(0.1)    # let it leave before the disconnect
        except Exception:
            pass                            # the firmware stops on disconnect and after 500 ms anyway

    def _on_disconnect(self, _client=None) -> None:
        self._wake.set()

    def _report(self, state: str, detail: str) -> None:
        msg = (state, detail[:_MAX_DETAIL], self.device)
        if msg != self._last_report:
            self._last_report = msg
            self._report_cb(*msg)


# -- the link process -----------------------------------------------------------------------------
def _worker(target: str, cmd_conn, status_conn, backend, settings: LinkSettings) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)   # Ctrl-C is the app's; it closes us (final S) properly
    os.set_blocking(status_conn.fileno(), False)
    try:
        asyncio.run(_worker_main(target, cmd_conn, status_conn, backend, settings))
    except Exception as exc:
        print(f"[rover] link process failed: {exc!r}", file=sys.stderr)
        raise


async def _worker_main(target: str, cmd_conn, status_conn, backend, settings: LinkSettings) -> None:
    loop = asyncio.get_running_loop()

    def report(state: str, detail: str, device: str) -> None:
        print(f"[rover] {state}: {detail}", file=sys.stderr, flush=True)
        try:
            status_conn.send((state, detail, device))
        except (BlockingIOError, OSError):
            pass    # the app isn't reading (or is gone): status is best-effort

    if backend is None:
        try:
            import bleak  # noqa: F401
        except ImportError as exc:
            report("error", f"bleak is not installed ({exc}): pip install bleak", "")
            return
        backend = BleakBackend(settings.connect_timeout_s)
    link = RoverLink(target, backend, report, settings)

    def on_readable() -> None:
        try:
            while cmd_conn.poll():
                link.command(cmd_conn.recv_bytes().decode("ascii", "replace"))
        except (EOFError, OSError):   # the app closed us, or died: final S and out
            loop.remove_reader(cmd_conn.fileno())
            link.close()

    loop.add_reader(cmd_conn.fileno(), on_readable)
    loop.add_signal_handler(signal.SIGTERM, link.close)
    await link.run()


# -- the app's side -------------------------------------------------------------------------------
class MicrobitSink(Sink):
    """Sends the command stream to the real rover. ``send`` only writes a few bytes to a non-blocking
    pipe; the rover-link process does the Bluetooth. A crashed link process is restarted."""

    name = "rover"
    RESTART_S = 3.0

    def __init__(self, target: Optional[str] = None, *, backend=None,
                 settings: Optional[LinkSettings] = None) -> None:
        self.target = config.MICROBIT_TARGET if target is None else target
        self.settings = settings or LinkSettings.from_config()
        self._backend = backend          # None: bleak, in the link process
        self._ctx = mp.get_context("spawn")
        self._proc = None
        self._cmd_w = None
        self._status_r = None
        self._state: tuple[str, str, str] = ("starting", "starting the Bluetooth link", "")
        self._closed = False
        self._restart_at = 0.0
        self.last_command: Optional[str] = None
        self.dropped = 0                 # sends the link process wasn't there to read
        self._start()

    def _start(self) -> None:
        cmd_r, cmd_w = self._ctx.Pipe(duplex=False)
        status_r, status_w = self._ctx.Pipe(duplex=False)
        proc = self._ctx.Process(target=_worker, name="rover-link", daemon=True,
                                 args=(self.target, cmd_r, status_w, self._backend, self.settings))
        proc.start()
        cmd_r.close()                    # keep only our ends, so either side sees EOF when the other goes
        status_w.close()
        os.set_blocking(cmd_w.fileno(), False)   # send() must never wait for the link process
        self._proc, self._cmd_w, self._status_r = proc, cmd_w, status_r

    def _ensure_worker(self) -> None:
        if self._proc.is_alive() or time.monotonic() < self._restart_at:
            return
        print(f"[neurover] rover link process exited ({self._proc.exitcode}); restarting", file=sys.stderr)
        self._restart_at = time.monotonic() + self.RESTART_S
        for conn in (self._cmd_w, self._status_r):
            try:
                conn.close()
            except OSError:
                pass
        self._state = ("starting", "restarting the Bluetooth link", self._state[2])
        self._start()

    def send(self, command: str) -> None:
        commands.validate(command)
        self.last_command = command
        if self._closed:
            return
        self._ensure_worker()
        try:
            self._cmd_w.send_bytes(command.encode("ascii"))   # 5 bytes: one atomic write, or EAGAIN
        except BlockingIOError:
            self.dropped += 1            # link process stuck: it stops the rover itself (stale -> S)
        except OSError:
            pass                         # it died; restarted on a later send
        self._drain()

    def _drain(self) -> None:
        try:
            while self._status_r.poll():
                self._state = self._status_r.recv()
        except (EOFError, OSError):
            pass

    def _label(self) -> str:
        dev = self._state[2]
        return f"ROVER {dev}" if dev else "ROVER"

    def status(self) -> SinkStatus:
        self._drain()
        if self._closed:
            return SinkStatus(False, self._label(), "closed")
        if not self._proc.is_alive():
            return SinkStatus(False, self._label(), f"link process exited ({self._proc.exitcode}); restarting")
        state, detail, _ = self._state
        return SinkStatus(state == READY, self._label(), detail)

    @property
    def state(self) -> str:
        self._drain()
        return self._state[0]

    def close(self) -> None:
        """Final S (acknowledged), disconnect, end the link process. Waits up to MICROBIT_CLOSE_S."""
        if self._closed:
            return
        self._closed = True
        try:
            self._cmd_w.close()          # EOF: the link process sends S, disconnects and exits
        except OSError:
            pass
        self._proc.join(self.settings.close_s)
        if self._proc.is_alive():
            self._proc.terminate()       # SIGTERM: same final S, if it can still run
            self._proc.join(1.0)
        if self._proc.is_alive():
            self._proc.kill()
            self._proc.join(1.0)
        self._drain()
        try:
            self._status_r.close()
        except OSError:
            pass
