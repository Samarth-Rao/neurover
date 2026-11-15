"""Synthetic EEG for running and testing without a headset (main.py --source mock, scope.py --mock).

One virtual head. Every electrode gets pink background noise, a resting alpha rhythm (weaker while
the user attends a flicker box), 60 Hz mains, a DC offset and, while the virtual user stares at a box
(``gaze``), an SSVEP at its frequency plus a weaker second harmonic, strongest over the occipital
cortex. Random blinks and jaw clenches add artifacts (``jaw`` holds one). With ``motor=True``, mu +
beta rhythms sit over each hand area (C3, C4) and weaken on the side opposite the hand the virtual
subject moves or imagines moving (``hand``), for the scope's cued trials.

Like the real BrainBits, each device also adds its OWN reference noise to all of its channels: a slow
wander and, now and then, a pop when its reference electrode shifts. On the real Flex + headband that
shared part was 60-97% of the power, and it is what made every channel of a device look bad at once.
"""
from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence

import numpy as np
from scipy import signal

_PINK_B = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
_PINK_A = [1.0, -2.494956002, 2.017265875, -0.522189400]

# How strongly each position picks up the SSVEP / alpha (occipital strongest).
SPATIAL = {"O1": 0.85, "Oz": 1.0, "O2": 0.85, "Iz": 0.8, "PO3": 0.75, "PO1": 0.85, "POz": 0.9, "PO2": 0.85,
           "PO4": 0.75, "PO7": 0.6,
           "PO8": 0.6, "Pz": 0.55, "P3": 0.5, "P4": 0.5, "T5": 0.35, "T6": 0.35, "T3": 0.15, "T4": 0.15}

# "clean": drives steadily by hand. "realistic": weak SSVEP, random artifacts, reference pops.
PRESETS = {
    "clean": dict(ssvep_uv=6.0, alpha_uv=1.0, artifacts=False, ref_pops=False),
    "realistic": dict(ssvep_uv=2.4, alpha_uv=3.0, artifacts=True, ref_pops=True),
}
NOISE_UV = 10.0            # background (RMS)
MAINS_UV = 4.0
DC_UV = 300.0
BLINK_EVERY_S = 6.0
JAW_EVERY_S = 30.0
REF_WANDER_UV = 12.0       # each device's shared reference noise (RMS, mostly < 10 Hz)
REF_POP_EVERY_S = 8.0
REF_POP_UV = (60.0, 250.0)
MU_UV = 6.0                # motor rhythms (with motor=True): RMS, ~10.5 Hz
BETA_UV = 3.0              # RMS, ~21 Hz
ERD = 0.75                 # rhythm amplitude over the moving hand's motor cortex (about -2 dB)
ERD_IPSI = 0.95            # ... and over the other side
ERD_TAU_S = 0.4            # how fast it follows the cue


def _unit_rms_lfilter(b, a) -> float:
    impulse = np.zeros(20000)
    impulse[0] = 1.0
    return float(np.sqrt(np.sum(signal.lfilter(b, a, impulse) ** 2)))


def _unit_rms_sos(sos) -> float:
    impulse = np.zeros(20000)
    impulse[0] = 1.0
    return float(np.sqrt(np.sum(signal.sosfilt(sos, impulse) ** 2)))


class MockHead:
    def __init__(self, positions: Sequence[str], fs: float, targets: Mapping[str, float],
                 device_of: Optional[Sequence[int]] = None, seed: Optional[int] = None, *,
                 ssvep_uv: float = 2.4, alpha_uv: float = 3.0, artifacts: bool = True, ref_pops: bool = True,
                 motor: bool = False) -> None:
        self.rng = np.random.default_rng(seed)
        self.fs = fs
        self.positions = tuple(positions)
        self.targets = dict(targets)
        c = len(self.positions)
        self.device_of = np.asarray(device_of if device_of is not None else [0] * c)
        self.n_devices = int(self.device_of.max()) + 1 if c else 0
        self.weight = np.array([SPATIAL.get(p, 0.4) for p in self.positions])
        self.ssvep_uv, self.alpha_uv, self.artifacts, self.ref_pops = ssvep_uv, alpha_uv, artifacts, ref_pops
        self.gaze: Optional[str] = None
        self.jaw = False
        self.hand: Optional[str] = None            # "left" / "right": moving or imagining that hand
        self._motor = MotorRhythms(self.positions, fs, self.rng) if motor else None

        self._pink_zi = np.zeros((len(_PINK_A) - 1, c))
        self._pink_rms = _unit_rms_lfilter(_PINK_B, _PINK_A)
        self._alpha_sos = signal.butter(4, (8.5, 11.5), btype="bandpass", fs=fs, output="sos")
        self._alpha_zi = np.zeros((self._alpha_sos.shape[0], 2, 1))
        self._alpha_rms = _unit_rms_sos(self._alpha_sos)
        self._alpha_level = alpha_uv
        self._jaw_sos = signal.butter(4, [20.0, min(100.0, fs / 2 - 1)], btype="bandpass", fs=fs, output="sos")
        self._jaw_zi = np.zeros((self._jaw_sos.shape[0], 2, c))
        self._ref_sos = signal.butter(2, 8.0, btype="lowpass", fs=fs, output="sos")
        self._ref_zi = np.zeros((self._ref_sos.shape[0], 2, max(1, self.n_devices)))
        self._ref_rms = _unit_rms_sos(self._ref_sos)
        self._pop = np.zeros(max(1, self.n_devices))

        self.dc = self.rng.uniform(-1, 1, c) * DC_UV
        self.mains_gain = self.rng.uniform(0.5, 1.5, c)
        self.mains_phase = self.rng.uniform(0, 2 * math.pi)
        self.ssvep_phase = self.rng.uniform(0, 2 * math.pi, c) * 0.15
        self.env = {label: 0.0 for label in self.targets}
        self.n = 0
        self._events: list[tuple[int, int, str, float]] = []
        self._next = {"blink": self._draw(BLINK_EVERY_S), "jaw": self._draw(JAW_EVERY_S),
                      "pop": self._draw(REF_POP_EVERY_S, ref_pops)}
        self._manual_jaw = 0.0

    def _draw(self, mean_s: float, enabled: Optional[bool] = None) -> Optional[int]:
        on = self.artifacts if enabled is None else enabled
        if not on or mean_s <= 0:
            return None
        return self.n + int(self.rng.exponential(mean_s) * self.fs) + int(self.fs)

    def _ramp(self, start: float, target: float, n: int, tau_s: float) -> np.ndarray:
        k = np.arange(1, n + 1)
        return target + (start - target) * np.exp(-k / (tau_s * self.fs))

    def generate(self, n: int) -> np.ndarray:
        """The next ``n`` samples, µV, one column per position."""
        c = len(self.positions)
        if n <= 0:
            return np.zeros((0, c))
        fs, w = self.fs, self.weight
        t = (self.n + np.arange(n)) / fs

        pink, self._pink_zi = signal.lfilter(_PINK_B, _PINK_A, self.rng.normal(size=(n, c)), axis=0, zi=self._pink_zi)
        x = pink * (NOISE_UV / self._pink_rms)

        a_src, self._alpha_zi = signal.sosfilt(self._alpha_sos, self.rng.normal(size=(n, 1)), axis=0,
                                               zi=self._alpha_zi)
        level = self._ramp(self._alpha_level, self.alpha_uv * (0.5 if self.gaze else 1.0), n, 0.4)
        self._alpha_level = float(level[-1])
        x += (a_src[:, 0] / self._alpha_rms * level)[:, None] * w[None, :]

        for label, freq in self.targets.items():
            env = self._ramp(self.env[label], 1.0 if self.gaze == label else 0.0, n, 0.3)
            self.env[label] = float(env[-1])
            if env.max() < 1e-3:
                continue
            ph = 2 * math.pi * freq * t[:, None] + self.ssvep_phase[None, :]
            x += self.ssvep_uv * env[:, None] * (np.sin(ph) + 0.5 * np.sin(2 * ph + 0.7)) * w[None, :]

        x += MAINS_UV * np.sin(2 * math.pi * 60.0 * t + self.mains_phase)[:, None] * self.mains_gain[None, :]
        x += self.dc[None, :]
        x += self._artifacts(n)
        if self._motor is not None:
            x += self._motor.generate(n, self.hand)
        if self.n_devices:
            x += self._reference(n)[:, self.device_of]
        self.n += n
        return x

    def _reference(self, n: int) -> np.ndarray:
        """Each device's shared reference noise: wander (< 8 Hz) + decaying pops."""
        d = self.n_devices
        wander, self._ref_zi = signal.sosfilt(self._ref_sos, self.rng.normal(size=(n, d)), axis=0, zi=self._ref_zi)
        out = wander * (REF_WANDER_UV / self._ref_rms)
        end = self.n + n
        nxt = self._next["pop"]
        pops = np.zeros((n, d))
        if nxt is not None and nxt < end:
            k = int(self.rng.integers(d))
            pops[nxt - self.n, k] = self.rng.choice((-1, 1)) * self.rng.uniform(*REF_POP_UV)
            self._next["pop"] = self._draw(REF_POP_EVERY_S, self.ref_pops)
        decay = math.exp(-1.0 / (0.3 * self.fs))
        level, self._pop = signal.lfilter([1.0], [1.0, -decay], pops, axis=0, zi=self._pop[None, :] * decay)
        self._pop = level[-1]
        return out + level

    def _artifacts(self, n: int) -> np.ndarray:
        fs, c = self.fs, len(self.positions)
        end = self.n + n
        if self._next["blink"] is not None and self._next["blink"] < end:
            s = self._next["blink"]
            self._events.append((s, s + int(0.5 * fs), "blink", float(self.rng.uniform(20, 160))))
            self._next["blink"] = self._draw(BLINK_EVERY_S)
        if self._next["jaw"] is not None and self._next["jaw"] < end:
            s = self._next["jaw"]
            self._events.append((s, s + int(self.rng.uniform(0.6, 1.2) * fs), "jaw", float(self.rng.uniform(80, 150))))
            self._next["jaw"] = self._draw(JAW_EVERY_S)
        out = np.zeros((n, c))
        manual = self._ramp(self._manual_jaw, 1.0 if self.jaw else 0.0, n, 0.05)
        self._manual_jaw = float(manual[-1])
        jaw_env = manual if manual.max() > 1e-3 else np.zeros(n)
        jaw_amp = 120.0 if manual.max() > 1e-3 else 0.0
        frontal = np.array([1.6 if p.startswith(("Fp", "F")) else 1.4 if p == "Pz" else 0.8 for p in self.positions])
        temporal = np.array([1.0 if p in ("T3", "T4", "T5", "T6") else 0.5 for p in self.positions])
        for start, stop, kind, amp in self._events:
            if stop <= self.n or start >= end:
                continue
            k = np.arange(max(start, self.n), min(stop, end))
            if kind == "blink":
                centre = (start + stop) / 2
                out[k - self.n] += (amp * np.exp(-(((k - centre) / fs) / 0.07) ** 2))[:, None] * frontal[None, :]
            else:
                rise = np.clip(np.minimum(k - start, stop - k) / (0.05 * fs), 0, 1)
                jaw_env[k - self.n] = np.maximum(jaw_env[k - self.n], rise)
                jaw_amp = max(jaw_amp, amp)
        emg, self._jaw_zi = signal.sosfilt(self._jaw_sos, self.rng.normal(size=(n, c)), axis=0, zi=self._jaw_zi)
        if jaw_amp > 0:
            out += emg / (np.std(emg) + 1e-9) * jaw_amp * jaw_env[:, None] * temporal[None, :]
        self._events = [e for e in self._events if e[1] > end]
        return out


class MotorRhythms:
    """Mu + beta over each hemisphere's hand area: the left hemisphere at C3, the right at C4 (without
    those, the first two leads). Moving or imagining one hand weakens the rhythm on the opposite side
    (ERD). The rhythms are steady oscillations whose frequency wanders a little, so a trial's ERD isn't
    lost in their own bursts."""

    RHYTHMS = (10.5, 21.0)
    WANDER_HZ = 0.7

    def __init__(self, positions: Sequence[str], fs: float, rng: np.random.Generator) -> None:
        self.fs = fs
        self.rng = rng
        positions = list(positions)
        # mix[h, col]: how much hemisphere h (0 = left/C3, 1 = right/C4) shows at each column
        self.mix = np.zeros((2, len(positions)))
        if "C3" in positions and "C4" in positions:
            self.mix[0, positions.index("C3")] = self.mix[1, positions.index("C4")] = 1.0
        elif len(positions) >= 2:
            self.mix[0, 0] = self.mix[1, 1] = 1.0
        self.amp = np.sqrt(2) * np.array([MU_UV, BETA_UV])[:, None]
        self.f0 = np.array(self.RHYTHMS)[:, None]
        self.phase = rng.uniform(0, 2 * np.pi, (len(self.RHYTHMS), 2))
        self.a = math.exp(-1.0 / fs)                 # the frequency wanders with ~1 s memory
        self.wander_zi = np.zeros((1, len(self.RHYTHMS) * 2))
        self.gain = np.ones(2)

    def generate(self, n: int, hand: Optional[str]) -> np.ndarray:
        target = np.ones(2)
        if hand in ("left", "right"):
            contra = 1 if hand == "left" else 0
            target[contra], target[1 - contra] = ERD, ERD_IPSI
        k = np.arange(1, n + 1)[:, None]
        gain = target + (self.gain - target) * np.exp(-k / (ERD_TAU_S * self.fs))
        self.gain = gain[-1].copy()
        r = len(self.RHYTHMS)
        noise = self.rng.normal(size=(n, r * 2)) * self.WANDER_HZ * math.sqrt(1 - self.a ** 2)
        wander, self.wander_zi = signal.lfilter([1.0], [1.0, -self.a], noise, axis=0, zi=self.wander_zi)
        freq = self.f0[None] + wander.reshape(n, r, 2)                         # [n, rhythm, hemisphere]
        phase = self.phase[None] + 2 * np.pi * np.cumsum(freq, axis=0) / self.fs
        self.phase = phase[-1] % (2 * np.pi)
        src = np.sum(self.amp[None] * np.sin(phase), axis=1)                   # [n, hemisphere]
        return (src * gain) @ self.mix
