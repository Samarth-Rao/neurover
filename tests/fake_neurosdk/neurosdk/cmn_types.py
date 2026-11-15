"""Subset of neurosdk.cmn_types used by neurover, with the same names and values."""
from dataclasses import dataclass
from enum import Enum


class SensorFamily(Enum):
    LEBrainBit = 3
    LEBrainBitFlex = 20


class SensorCommand(Enum):
    StartSignal = 0
    StopSignal = 1
    StartResist = 2
    StopResist = 3


class SensorState(Enum):
    StateInRange = 0
    StateOutOfRange = 1


class SensorGain(Enum):
    Gain3 = 2


class BrainBit2ChannelMode(Enum):
    ChModeNormal = 1


class GenCurrent(Enum):
    GenCurr6nA = 1


@dataclass
class SensorInfo:
    SensFamily: SensorFamily
    SensModel: int
    Name: str
    Address: str
    SerialNumber: str
    PairingRequired: bool
    RSSI: int


@dataclass
class SensorVersion:
    FwMajor: int
    FwMinor: int
    FwPatch: int
    HwMajor: int
    HwMinor: int
    HwPatch: int
    ExtMajor: int


@dataclass
class EEGChannelInfo:
    Id: int
    ChType: int
    Name: str
    Num: int


@dataclass
class SignalChannelsData:
    PackNum: int
    Marker: int
    Samples: list


@dataclass
class ResistRefChannelsData:
    PackNum: int
    Samples: list
    Referents: list


@dataclass
class BrainBitSignalData:
    PackNum: int
    Marker: int
    O1: float
    O2: float
    T3: float
    T4: float


@dataclass
class BrainBitResistData:
    O1: float
    O2: float
    T3: float
    T4: float


@dataclass
class BrainBit2AmplifierParam:
    ChSignalMode: list
    ChResistUse: list
    ChGain: list
    Current: GenCurrent
