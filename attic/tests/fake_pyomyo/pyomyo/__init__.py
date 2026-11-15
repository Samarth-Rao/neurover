"""Test stand-in for pyomyo, with the same API surface neurorover uses (see sources/myo_worker.py).

Behaviour is chosen by the FAKE_MYO_SCENARIO environment variable, which the
spawned worker process inherits:
    stream      dongle + armband found, 50 Hz EMG forever
    no_dongle   Myo() raises ValueError('Myo dongle not found!') like the real one
    busy        opening the dongle's serial port fails
    drop        streams for ~1 s, then the armband goes silent (reads block)
"""
import enum
import os
import time


class emg_mode(enum.Enum):
    NO_DATA = 0
    PREPROCESSED = 1
    FILTERED = 2
    RAW = 3


class _Serial:
    def __init__(self, port):
        self.port = port


class _BT:
    def __init__(self, port):
        self.ser = _Serial(port)


class Myo:
    def __init__(self, tty=None, mode=1):
        self.scenario = os.environ.get("FAKE_MYO_SCENARIO", "stream")
        if self.scenario == "no_dongle":
            raise ValueError("Myo dongle not found!")
        if self.scenario == "busy":
            raise OSError("could not open port 'COM9': PermissionError(13, 'Access is denied.')")
        if mode is not emg_mode.PREPROCESSED:
            raise AssertionError(f"expected emg_mode.PREPROCESSED, got {mode!r}")
        self.bt = _BT(tty or "COM9")
        self.emg_handlers = []
        self.battery_handlers = []
        self.n = 0

    def add_emg_handler(self, h):
        self.emg_handlers.append(h)

    def add_battery_handler(self, h):
        self.battery_handlers.append(h)

    def connect(self, addr=None):
        for h in self.battery_handlers:
            h(87)

    def set_leds(self, logo, line):
        pass

    def vibrate(self, length):
        pass

    def run(self):
        time.sleep(0.02)  # 50 Hz
        if self.scenario == "drop" and self.n >= 50:
            time.sleep(3600)  # armband gone: the real serial read just blocks
        self.n += 1
        for h in self.emg_handlers:
            h((100, 200, 300, 400, 500, 600, 700, 800), 0)

    def disconnect(self):
        pass
