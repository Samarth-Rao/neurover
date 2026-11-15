"""Bluetooth latency to the real rover, measured. Only ever sends S (stop) and P (ping): the rover never moves.

    python tools/microbit_latency.py              # the first micro:bit seen
    python tools/microbit_latency.py tegip        # that one (part of its name)
    python tools/microbit_latency.py --scan       # list the micro:bits in range

Prints the time to find and to connect, then
  write+response   "S" written with response: the round trip to the micro:bit's Bluetooth stack.
                   Steps of the connection interval (macOS picks it: typically 15 or 30 ms).
  write, no resp.  "S" written without response: only the time to hand it to macOS (it leaves at the
                   next connection event, like a write with response does).
  ping             "P" -> the firmware's UART handler -> "P" back: the whole path through the firmware,
                   both ways. Needs firmware/maqueen.ts; the old firmware doesn't answer.
The rover link in the app (rover/microbit.py) writes without response, so one way to the micro:bit's
handler is about half a ping.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rover.microbit import UART_RX, UART_TX, PING, BleakBackend, matches, short_name  # noqa: E402

STOP = b"S\n"


def summary(name: str, seconds: list[float]) -> str:
    if not seconds:
        return f"{name:16s} -"
    ms = sorted(s * 1000 for s in seconds)
    q = lambda p: ms[min(len(ms) - 1, int(p * len(ms)))]  # noqa: E731
    return (f"{name:16s} n={len(ms):3d}  min {ms[0]:6.1f}  median {q(0.5):6.1f}  p95 {q(0.95):6.1f}  "
            f"max {ms[-1]:6.1f} ms")


async def scan(seconds: float) -> int:
    from bleak import BleakScanner
    seen: dict[str, tuple[str, int]] = {}

    def on_adv(d, ad) -> None:
        name = ad.local_name or d.name
        if matches("", d.address, name):
            seen[d.address] = (name, ad.rssi)

    print(f"scanning {seconds:.0f} s for micro:bits…")
    async with BleakScanner(on_adv):
        await asyncio.sleep(seconds)
    for addr, (name, rssi) in sorted(seen.items(), key=lambda kv: -kv[1][1]):
        print(f"  {short_name(name):8s} {name:24s} {addr}  rssi {rssi}")
    if not seen:
        print("  none (a micro:bit that is connected to something doesn't advertise)")
    return 0 if seen else 1


async def measure(target: str, n: int, backend=None, pace_s: float = 0.1) -> int:
    backend = backend or BleakBackend()
    t0 = time.perf_counter()
    device = await backend.find(target, 15.0)
    if device is None:
        print(f"{target or 'no micro:bit'} not found")
        return 1
    print(f"found {device.name} ({device.address}) in {time.perf_counter() - t0:.1f} s")
    client = backend.client(device, lambda _c: None)
    t0 = time.perf_counter()
    await client.connect()
    print(f"connected in {time.perf_counter() - t0:.1f} s")
    results = []
    try:
        rx = client.services.get_characteristic(UART_RX)
        tx = client.services.get_characteristic(UART_TX)
        if rx is None:
            print("no UART service: is the firmware's bluetooth.startUartService() running?")
            return 1
        print(f"UART RX properties: {', '.join(rx.properties)}; MTU {getattr(client, 'mtu_size', '?')}")

        rtt = []
        for _ in range(n):
            t = time.perf_counter()
            await client.write_gatt_char(rx, STOP, response=True)
            rtt.append(time.perf_counter() - t)
            await asyncio.sleep(pace_s)
        results.append(summary("write+response", rtt))

        if "write-without-response" in rx.properties:
            call = []
            for _ in range(n):
                t = time.perf_counter()
                await client.write_gatt_char(rx, STOP, response=False)
                call.append(time.perf_counter() - t)
                await asyncio.sleep(pace_s)
            results.append(summary("write, no resp.", call))

        pings = []
        if tx is not None:
            got = asyncio.Event()
            buf = bytearray()

            def on_data(_c, data) -> None:
                buf.extend(data)
                if PING in buf:
                    buf.clear()
                    got.set()

            await client.start_notify(tx, on_data)
            for _ in range(n):
                got.clear()
                t = time.perf_counter()
                await client.write_gatt_char(rx, PING, response=False)
                try:
                    await asyncio.wait_for(got.wait(), 2.0)
                except asyncio.TimeoutError:
                    print("no answer to P: the old firmware (flash firmware/maqueen.ts to measure the ping)")
                    break
                pings.append(time.perf_counter() - t)
                await asyncio.sleep(pace_s)
        results.append(summary("ping", pings))
    finally:
        try:
            await client.write_gatt_char(rx, STOP, response=True)
        except Exception:
            pass
        await client.disconnect()
    print("\n".join(results))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("target", nargs="?", default="", help="part of the micro:bit's name, or its address")
    p.add_argument("--scan", action="store_true", help="just list the micro:bits in range")
    p.add_argument("-n", type=int, default=30, help="samples per measurement (default 30)")
    args = p.parse_args(argv)
    if args.scan:
        return asyncio.run(scan(6.0))
    return asyncio.run(measure(args.target, args.n))


if __name__ == "__main__":
    sys.exit(main())
