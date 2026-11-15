"""Child-process side of the Myo source over the computer's own Bluetooth (no Bluegiga dongle), via bleak.

Myo Bluetooth protocol, from Thalmic Labs' published myohw.h. Every Myo UUID is
d506XXXX-a904-deb9-4748-2c7f4a124842:

    0x0001  control service (the armband advertises it)
    0x0401  command characteristic (write): [command, payload size, payload...]
              set_mode        0x01: [0x01, 3, emg_mode, imu_mode, classifier_mode]
                              emg_mode 0x02 = filtered EMG (0x03 raw, 0x00 off)
              set_sleep_mode  0x09: [0x09, 1, 1] = never sleep (1) / normal (0)
              vibrate         0x03: [0x03, 1, 1] = short buzz
    0x0105, 0x0205, 0x0305, 0x0405  EMG data (notify): 16 bytes = 2 samples x 8 channels of int8;
              each characteristic notifies at 50 Hz, so 200 Hz in total
    battery: the standard 0x2a19 characteristic (read + notify)

If the armband won't show up: it may be in deep sleep. Plug its USB cable in for a moment to
wake it, then unplug it (while charging it doesn't stream).

Same messages as myo_worker.py, so MyoSource doesn't care which one runs:
    ("scanning", where) ("connected", where) ("emg", t, (v0..v7)) ("battery", %) ("error", message)
"""
from __future__ import annotations

import asyncio
import struct
import time
from typing import Callable, Optional

BASE = "d506{:04x}-a904-deb9-4748-2c7f4a124842"
CONTROL_SERVICE = BASE.format(0x0001)
COMMAND = BASE.format(0x0401)
EMG_CHARS = tuple(BASE.format(n) for n in (0x0105, 0x0205, 0x0305, 0x0405))
BATTERY = "00002a19-0000-1000-8000-00805f9b34fb"

SET_MODE_FILTERED_EMG = bytes([0x01, 3, 0x02, 0x00, 0x00])
SET_MODE_OFF = bytes([0x01, 3, 0x00, 0x00, 0x00])
NEVER_SLEEP = bytes([0x09, 1, 1])
NORMAL_SLEEP = bytes([0x09, 1, 0])
VIBRATE_SHORT = bytes([0x03, 1, 1])
FS = 200.0
SCAN_S = 4.0


class SampleClock:
    """Timestamps for a 200 Hz stream that arrives in bursts: count samples at the device rate and
    slew gently towards arrival time (as brainbit_worker does), so windows never see time jumps."""

    def __init__(self, fs: float = FS) -> None:
        self.dt = 1.0 / fs
        self.next_t: Optional[float] = None

    def stamp(self, n: int, now: float) -> list[float]:
        if self.next_t is None:
            self.next_t = now - (n - 1) * self.dt
        error = now - (self.next_t + (n - 1) * self.dt)
        if abs(error) > 0.5:
            self.next_t += error
        elif error > 0:
            self.next_t += 0.05 * error
        out = [self.next_t + i * self.dt for i in range(n)]
        self.next_t = out[-1] + self.dt
        return out


def parse_emg(data: bytes) -> list[tuple[int, ...]]:
    """One EMG notification -> its two 8-channel samples."""
    if len(data) < 16:
        return []
    return [struct.unpack("<8b", bytes(data[0:8])), struct.unpack("<8b", bytes(data[8:16]))]


def is_myo(device, adv, name_hint: Optional[str] = None) -> bool:
    name = (getattr(adv, "local_name", None) or getattr(device, "name", None) or "")
    if name_hint:
        return name_hint.lower() in name.lower() or name_hint.lower() in str(device.address).lower()
    uuids = [u.lower() for u in (getattr(adv, "service_uuids", None) or [])]
    return CONTROL_SERVICE in uuids or "myo" in name.lower()


def run(q, stop, name_hint, log_path) -> None:
    def put(*msg) -> None:
        try:
            q.put(msg)
        except Exception:
            pass

    try:
        import bleak  # noqa: F401
    except ImportError as exc:
        put("error", f"bleak is not installed ({exc}). Run: pip install bleak")
        return
    try:
        asyncio.run(_main(put, stop, name_hint))
    except Exception as exc:
        put("error", f"Myo Bluetooth: {exc}")


async def _main(put: Callable, stop, name_hint: Optional[str]) -> None:
    from bleak import BleakClient, BleakScanner

    put("scanning", "Bluetooth")
    device = None
    while device is None:
        if stop.is_set():
            return
        device = await BleakScanner.find_device_by_filter(lambda d, a: is_myo(d, a, name_hint), timeout=SCAN_S)

    lost = asyncio.Event()
    clock = SampleClock()

    def on_emg(_char, data: bytearray) -> None:
        samples = parse_emg(data)
        if samples:
            for t, values in zip(clock.stamp(len(samples), time.monotonic()), samples):
                put("emg", t, values)

    async with BleakClient(device, disconnected_callback=lambda _c: lost.set(), timeout=20.0) as client:
        await client.write_gatt_char(COMMAND, NEVER_SLEEP, response=True)
        await client.write_gatt_char(COMMAND, SET_MODE_FILTERED_EMG, response=True)
        for uuid in EMG_CHARS:
            await client.start_notify(uuid, on_emg)
        put("connected", f"Bluetooth ({device.name or device.address})")
        try:
            level = await client.read_gatt_char(BATTERY)
            put("battery", int(level[0]))
            await client.start_notify(BATTERY, lambda _c, d: put("battery", int(d[0])))
        except Exception:
            pass
        try:
            await client.write_gatt_char(COMMAND, VIBRATE_SHORT, response=True)
        except Exception:
            pass
        while not stop.is_set() and not lost.is_set():
            await asyncio.sleep(0.1)
        if lost.is_set():
            put("error", "Myo Bluetooth connection lost.")
            return
        for cmd in (SET_MODE_OFF, NORMAL_SLEEP):  # leave the armband as we found it
            try:
                await client.write_gatt_char(COMMAND, cmd, response=True)
            except Exception:
                pass
