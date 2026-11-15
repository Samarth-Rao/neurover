"""Fake neurosdk.scanner. Scenario from FAKE_BRAINBIT:
    flex4 | flex1 | none | badcontact | drop     one device
    dual                                          a Flex 4 and a BrainBit headband
    dual-drop                                     both; the headband's Bluetooth drops once after ~3 s
    dual-v1                                       a Flex v1 + a headband: same family, same name
"""
import math
import os
import threading
import time

from neurosdk.cmn_types import (BrainBitResistData, BrainBitSignalData, EEGChannelInfo, ResistRefChannelsData,
                                SensorCommand, SensorFamily, SensorInfo, SensorState, SensorVersion,
                                SignalChannelsData)

FS = 250.0
# Flex 4 reports its channels in an order unlike config (mapping must go by name).
FLEX4_CHANNELS = [("T3", 0), ("O1", 1), ("T4", 2), ("O2", 3)]


def _value(name, t):
    """Volts. O1 carries a big 5 Hz marker so tests can see where it lands."""
    if name == "O1":
        return 40e-6 * math.sin(2 * math.pi * 5 * t)
    return 3e-6 * math.sin(2 * math.pi * 11 * t + len(name))


def _headband_value(name, t):
    """Volts. The headband's O2 carries a 30 µV 7 Hz marker (unlike the Flex's O1 marker)."""
    if name == "O2":
        return 30e-6 * math.sin(2 * math.pi * 7 * t)
    return 2e-6 * math.sin(2 * math.pi * 13 * t + len(name))


class _FakeSensor:
    def __init__(self, family, scenario):
        self.sens_family = family
        self.scenario = scenario
        self.state = SensorState.StateInRange
        self.batt_power = 87
        self.signalDataReceived = None
        self.resistDataReceived = None
        self.sensorStateChanged = None
        self.batteryChanged = None
        self.commands = []
        self.connect_attempts = 0
        self._mode = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self._n = 0
        self._streamed_s = 0.0
        self._dropped = False
        self._t0 = None

    def exec_command(self, cmd):
        self.commands.append(cmd)
        if cmd == SensorCommand.StartSignal:
            self._mode = "signal"
        elif cmd == SensorCommand.StartResist:
            self._mode = "resist"
        elif cmd in (SensorCommand.StopSignal, SensorCommand.StopResist):
            self._mode = None

    def connect(self):
        self.connect_attempts += 1
        if self.connect_attempts < 4:  # ~1 s outage with the tests' 0.3 s retry interval
            raise Exception("Code 110. Message:Failed to connect the sensor")
        self.state = SensorState.StateInRange
        if self.sensorStateChanged:
            self.sensorStateChanged(self, self.state)

    def disconnect(self):
        self._stop.set()
        self.state = SensorState.StateOutOfRange

    def _loop(self):
        while not self._stop.is_set():
            time.sleep(0.02)
            if self.state != SensorState.StateInRange:
                self._t0 = None
                continue
            if self._mode != "signal":
                self._t0 = None
            if self._mode == "signal":
                if self.scenario == "drop" and not self._dropped and self._streamed_s > 3.0:
                    self._dropped = True
                    self.state = SensorState.StateOutOfRange
                    self._mode = None
                    if self.sensorStateChanged:
                        self.sensorStateChanged(self, self.state)
                    continue
                # Like real hardware: the device clock sets the rate (250 Hz), whatever our sleep jitter.
                now = time.monotonic()
                if self._t0 is None:
                    self._t0, self._sent = now, 0
                due = int((now - self._t0) * FS) - self._sent
                packet = []
                for _ in range(max(0, due)):
                    t = self._n / FS
                    self._n += 1
                    packet.append(self._signal_sample(t))
                self._sent += len(packet)
                self._streamed_s += len(packet) / FS
                if self.signalDataReceived:
                    self.signalDataReceived(self, packet)
            elif self._mode == "resist" and self.resistDataReceived:
                self.resistDataReceived(self, self._resist_sample())

    def _ohms(self, name):
        if self.scenario == "badcontact" and name == "T4":
            return 5e6
        return 3e5


class _FakeFlex4(_FakeSensor):
    def __init__(self, scenario):
        super().__init__(SensorFamily.LEBrainBitFlex, scenario)
        self.supported_channels = [EEGChannelInfo(Id=0, ChType=0, Name=n, Num=i) for n, i in FLEX4_CHANNELS]
        self.amplifier_param = None

    def _signal_sample(self, t):
        samples = [0.0] * 4
        for name, num in FLEX4_CHANNELS:
            samples[num] = _value(name, t)
        return SignalChannelsData(PackNum=self._n, Marker=0, Samples=samples)

    def _resist_sample(self):
        samples = [0.0] * 4
        for name, num in FLEX4_CHANNELS:
            samples[num] = self._ohms(name)
        return [ResistRefChannelsData(PackNum=0, Samples=samples, Referents=[0.0, 0.0])]


class _FakeFlex1(_FakeSensor):
    def __init__(self, scenario):
        super().__init__(SensorFamily.LEBrainBit, scenario)
        self.version = SensorVersion(FwMajor=101, FwMinor=0, FwPatch=0, HwMajor=1, HwMinor=0, HwPatch=0, ExtMajor=0)
        self.gain = None

    def _signal_sample(self, t):
        return BrainBitSignalData(PackNum=self._n % 2048, Marker=0,
                                  **{n: _value(n, t) for n in ("O1", "O2", "T3", "T4")})

    def _resist_sample(self):
        return BrainBitResistData(**{n: self._ohms(n) for n in ("O1", "O2", "T3", "T4")})


class _FakeHeadband(_FakeFlex1):
    """The classic BrainBit headband: same family as Flex v1, but firmware major < 100."""

    def __init__(self, scenario):
        super().__init__(scenario)
        self.version = SensorVersion(FwMajor=3, FwMinor=0, FwPatch=0, HwMajor=1, HwMinor=0, HwPatch=0, ExtMajor=0)
        self.batt_power = 64

    def _signal_sample(self, t):
        return BrainBitSignalData(PackNum=self._n % 2048, Marker=0,
                                  **{n: _headband_value(n, t) for n in ("O1", "O2", "T3", "T4")})


class Scanner:
    last = None

    def __init__(self, filters):
        self.filters = list(filters)
        self.sensorsChanged = None
        self.scenario = os.environ.get("FAKE_BRAINBIT", "flex4")
        self.created = []
        Scanner.last = self

    def start(self):
        pass

    def stop(self):
        pass

    def sensors(self):
        if self.scenario == "none":
            return []
        if self.scenario == "dual-v1":
            return [SensorInfo(SensFamily=SensorFamily.LEBrainBit, SensModel=0, Name="BrainBit", Address=a,
                               SerialNumber=sn, PairingRequired=False, RSSI=rssi) for a, sn, rssi in
                    [(a, sn, rssi) for a, sn, rssi in (("FLEX-V1", "450904", -70), ("HEAD", "133013", -40))
                     if SensorFamily.LEBrainBit in self.filters]]
        if self.scenario.startswith("dual"):
            fams = [SensorFamily.LEBrainBitFlex, SensorFamily.LEBrainBit]
        else:
            fams = [SensorFamily.LEBrainBit if self.scenario == "flex1" else SensorFamily.LEBrainBitFlex]
        out = []
        for i, fam in enumerate(fams):
            if fam not in self.filters:
                continue
            name = "BrainBit" if fam == SensorFamily.LEBrainBit else "Flex"
            out.append(SensorInfo(SensFamily=fam, SensModel=2, Name=name, Address=f"AA-BB-{i}",
                                  SerialNumber=f"12345{i}", PairingRequired=False, RSSI=-60))
        return out

    def create_sensor(self, info):
        if self.scenario == "dual-v1":
            s = _FakeFlex1("ok") if info.Address == "FLEX-V1" else _FakeHeadband("ok")
            self.created.append(s)
            return s
        if info.SensFamily == SensorFamily.LEBrainBit:
            if self.scenario.startswith("dual"):
                s = _FakeHeadband("drop" if self.scenario == "dual-drop" else "ok")
            else:
                s = _FakeFlex1(self.scenario)
        else:
            s = _FakeFlex4("ok" if self.scenario.startswith("dual") else self.scenario)
        self.created.append(s)
        return s
