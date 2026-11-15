"""Read-only BrainBit diagnostic: what does the connected headband actually report?

    python tools/brainbit_probe.py [seconds]

Scans, connects to the strongest BrainBit/Flex, prints its family/model/firmware,
channel names and order, battery, ~3 s of electrode resistance and a few seconds
of signal statistics, then disconnects. Use it to fill in config.EEG_SDK_CHANNELS
and EEG_CHANNEL_MAP, or to debug a headband that won't stream.
"""
from __future__ import annotations

import math
import sys
import threading
import time

import numpy as np

from neurosdk.cmn_types import SensorCommand, SensorFamily
from neurosdk.scanner import Scanner


def main(seconds: float = 6.0) -> int:
    scanner = Scanner([SensorFamily.LEBrainBit, SensorFamily.LEBrainBitFlex])
    scanner.start()
    print("scanning (15 s max)…")
    t0, infos = time.monotonic(), []
    while time.monotonic() - t0 < 15 and not infos:
        time.sleep(0.5)
        infos = scanner.sensors()
    scanner.stop()
    if not infos:
        print("no BrainBit found")
        return 1
    for i in infos:
        print(f"found: {i}")
    info = max(infos, key=lambda i: i.RSSI)
    print(f"connecting to {info.Name} {info.SerialNumber}…")
    sensor = scanner.create_sensor(info)
    try:
        fam = sensor.sens_family
        print(f"family={fam}  name={sensor.name}  battery={sensor.batt_power}")
        try:
            print(f"version={sensor.version}")
        except Exception as exc:
            print(f"version: n/a ({exc})")
        try:
            print(f"sampling_frequency={sensor.sampling_frequency}")
        except Exception as exc:
            print(f"sampling_frequency: n/a ({exc})")
        if fam == SensorFamily.LEBrainBitFlex:
            print(f"channels_count={sensor.channels_count}")
            for ch in sensor.supported_channels:
                print(f"  channel {ch}")
            try:
                print(f"amplifier_param={sensor.amplifier_param}")
            except Exception as exc:
                print(f"amplifier_param: unreadable ({exc})")

        lock = threading.Lock()
        resist, packets, samples = [], [], []

        def on_resist(_s, data):
            with lock:
                resist.extend(data if isinstance(data, list) else [data])

        def on_signal(_s, data):
            with lock:
                packets.append((time.monotonic(), len(data)))
                samples.extend(data)

        sensor.resistDataReceived = on_resist
        sensor.signalDataReceived = on_signal

        sensor.exec_command(SensorCommand.StartResist)
        time.sleep(3.0)
        sensor.exec_command(SensorCommand.StopResist)
        with lock:
            print(f"resistance callbacks: {len(resist)}; last: {resist[-1] if resist else None}")

        sensor.exec_command(SensorCommand.StartSignal)
        time.sleep(seconds)
        sensor.exec_command(SensorCommand.StopSignal)
        with lock:
            n = len(samples)
            if not n:
                print("no signal samples received")
                return 1
            span = packets[-1][0] - packets[0][0]
            sizes = sorted({p[1] for p in packets})
            print(f"signal: {n} samples in {len(packets)} callbacks over {span:.2f} s "
                  f"(~{n / max(span, 1e-6):.0f} Hz); samples per callback: {sizes}")
            print(f"first sample: {samples[0]}")
            if hasattr(samples[0], "Samples"):
                x = np.array([s.Samples for s in samples], dtype=float)
            else:
                x = np.array([[s.O1, s.O2, s.T3, s.T4] for s in samples], dtype=float)
            x_uv = x * 1e6
            print("per channel (assuming volts): mean µV", np.round(x_uv.mean(0), 1),
                  " std µV", np.round(x_uv.std(0), 1))
            packs = [s.PackNum for s in samples]
            gaps = sum(1 for a, b in zip(packs, packs[1:]) if b != a + 1 and not (b == 0))
            print(f"PackNum range {min(packs)}..{max(packs)}; non-consecutive steps: {gaps}")
    finally:
        sensor.resistDataReceived = None
        sensor.signalDataReceived = None
        try:
            sensor.disconnect()
        except Exception:
            pass
        del sensor
        del scanner
        print("disconnected")
    return 0


if __name__ == "__main__":
    sys.exit(main(float(sys.argv[1]) if len(sys.argv) > 1 else 6.0))
