"""Shared pipeline for EEG sources (mock EEG, BrainBit Flex + BrainBit headband).

The electrodes come from one or more devices (config.EEG_DEVICES), each with its own SDK
channel names and a map to where each electrode sits on the head. Subclasses provide raw
chunks with every device's SDK channels side by side (device order, then SDK order), in µV:
``_read_chunk() -> (t[n], x[n, sdk_channels])`` or ``(t, x, present[n, devices])`` when a
device can be missing for a while. This class then

    remaps SDK channels -> physical positions
    -> streaming filters (SSVEP band + mains notch; 1-40 Hz for display/artifacts)
    -> sliding windows (SSVEP_WINDOW_S, hop SSVEP_HOP_S)
    -> per device: its decoding electrodes, minus any the quality check flags as bad;
       an artifact (> EEG_ARTIFACT_UV) drops that device from the window. If no device is
       left the window is dropped (no prediction at all, so a sustained artifact trips the
       dead-man -> S; it can never become a command)
    -> decoder (SSVEP CCA per device, or alpha ratio) -> Prediction, with scores for the dashboard.

Decoders share a small duck-typed interface (pipeline/ssvep.py SsvepDecoder,
pipeline/alpha.py AlphaDecoder): classes, command_map, class_names, threshold,
evaluate(), calibration_segments(), calibrate(), save(), load().
"""
from __future__ import annotations

import abc
import time
import warnings
from collections import defaultdict, deque
from typing import Callable, Optional

import numpy as np

import config
from pipeline import alpha, ssvep, ssvep_cal
from pipeline.eeg import StreamFilter, channel_permutation, spectrum
from pipeline.features import Windower

from .base import Prediction, Source

MODES = ("ssvep", "alpha")
_ARROWS = {"up": "↑", "down": "↓", "left": "←", "right": "→"}


class EegStreamSource(Source):
    eeg = True
    supports_training = False
    # The decoders gate on their own (calibrated) threshold + margin, so the smoother's generic
    # probability threshold is switched off for EEG; its N-window debounce and dead-man still apply.
    # It must be -inf, not 0: calibrated scores (log-likelihood ratios, z-scores) are negative when
    # nothing is looked at, and a "low confidence" rest window would let the last drive command stick.
    confidence_threshold = float("-inf")

    def __init__(self, mode: str = "ssvep", *, devices=None, sdk_channels=None,
                 channel_map=None, fs: float = config.EEG_SAMPLE_HZ,
                 window_s: Optional[float] = None, hop_s: Optional[float] = None,
                 user: str = "default", models_dir=config.MODELS_DIR,
                 refresh_hz: float = config.SSVEP_REFRESH_HZ,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown EEG mode {mode!r}; choose from {', '.join(MODES)}")
        self._explicit_window = window_s is not None   # else a calibration may choose the window length
        if window_s is None:
            window_s = config.SSVEP_WINDOW_S if mode == "ssvep" else config.ALPHA_WINDOW_S
        if hop_s is None:
            hop_s = config.SSVEP_HOP_S if mode == "ssvep" else config.ALPHA_HOP_S
        if not 1.0 <= window_s <= 4.0:
            raise ValueError("EEG window must be between 1 and 4 seconds")
        self.mode = mode
        self.fs = fs
        self.clock = clock
        self.user = user
        self.models_dir = models_dir
        self.window_s = window_s
        self.devices = resolve_devices(devices, sdk_channels, channel_map)
        self.device_names = tuple(self.devices)
        self.device_labels = {n: d.get("label", n) for n, d in self.devices.items()}
        sdk_dev = [k for k, d in enumerate(self.devices.values()) for _ in d["map"]]
        self.sdk_channels = tuple(f"{n}:{ch}" for n, d in self.devices.items() for ch in d["map"])
        self.channel_map = {f"{n}:{ch}": pos for n, d in self.devices.items() for ch, pos in d["map"].items()}
        self._perm, positions = channel_permutation(self.sdk_channels, self.channel_map, config.EEG_DISPLAY_ORDER)
        self.positions = tuple(positions)
        self.channels = len(self.positions)
        self.device_of = [sdk_dev[j] for j in self._perm]           # device index of each position
        self.device_positions = {n: tuple(p for i, p in enumerate(self.positions) if self.device_of[i] == k)
                                 for k, n in enumerate(self.device_names)}
        self.occipital = [i for i, p in enumerate(self.positions) if p in config.EEG_OCCIPITAL]
        decode = [i for i, p in enumerate(self.positions) if p in config.EEG_DECODE_POSITIONS]
        self.decode_idx = decode or list(range(len(self.positions)))  # none configured present -> use all
        self.noise_idx = [i for i, p in enumerate(self.positions)
                          if p in config.EEG_NOISE_REF_POSITIONS and i not in self.decode_idx]
        self.layout = ssvep_cal.Layout(self.positions, self.device_names, self.device_of, self.decode_idx,
                                       self.noise_idx)

        self._ssvep_filter = StreamFilter(fs, config.SSVEP_BAND, config.EEG_NOTCH_HZ)
        self._display_filter = StreamFilter(fs, config.EEG_DISPLAY_BAND, config.EEG_NOTCH_HZ)
        self._muscle = ssvep_cal.MuscleDetector(fs, len(self.sdk_channels)) if mode == "ssvep" else None
        self.blanked = 0.0        # share of the latest window's samples patched over (worst device)
        # For the decoder view (I): per device, how much each box's frequency (and its double) stands
        # out from the neighbouring frequencies in the latest window (~1x = nothing), and recent scores.
        self.response: dict[str, dict[str, float]] = {}
        self.score_history: deque = deque(maxlen=max(10, int(10.0 / hop_s)))
        self.hop_s = hop_s
        # Tolerate some Bluetooth packet loss: a window needs 75% of its samples.
        self.windower = Windower(window_s, hop_s, min_samples=int(window_s * fs * 0.75))
        n_trace = int(config.EEG_TRACE_S * fs)
        self._trace = np.zeros((n_trace, self.channels))
        self._trace_pos = 0

        if mode == "ssvep":
            self.decoder = ssvep.SsvepDecoder(fs, ssvep.target_frequencies(config.SSVEP_FRAMES, refresh_hz),
                                              groups=self.device_names, setup=self.decode_setup(),
                                              positions=self.positions)
        else:
            self.decoder = alpha.AlphaDecoder(fs)
        self.classes = tuple(self.decoder.classes)
        self.command_map = dict(self.decoder.command_map)
        self.class_names = dict(self.decoder.class_names)
        # SSVEP: a calibration brings its own window length, unless one was asked for explicitly.
        want_window = window_s if (mode == "alpha" or self._explicit_window) else None
        self.calibration, self.load_message = self.decoder.load(user, want_window, models_dir)
        self._adopt_window(self.calibration)
        self._cal_log: list[tuple[str, float]] = []

        self.scores: dict[str, float] = {l: 0.0 for l in self.classes}
        self.spectrum: Optional[tuple[np.ndarray, np.ndarray]] = None
        self.artifact = False
        self.artifact_windows = 0
        self.clean_windows = 0
        # Per-position electrode contact from a resistance test: True good, False bad, None unknown.
        self.contact: dict[str, Optional[bool]] = {p: None for p in self.positions}
        # Continuous signal check: "unknown" | "good" | "bad", with a reason; latched (see _update_quality).
        self.quality: dict[str, str] = {p: "unknown" for p in self.positions}
        self.quality_note: dict[str, str] = {}
        self._amp_hist: deque = deque()
        self._good_since: dict[str, Optional[float]] = {p: None for p in self.positions}
        self._bad_signal: dict[str, bool] = {}   # bad because of the signal itself (not just resistance)
        self._raw_peak = np.zeros(len(self.positions))
        self._prev_present: Optional[np.ndarray] = None
        self.present_devices: tuple[str, ...] = ()    # devices with data in the latest window
        self.decoding: dict[str, list[str]] = {}      # device -> electrodes the latest window was decoded from
        self.device_artifact: set[str] = set()        # devices dropped from the latest window by an artifact
        self.calibrating: Optional[str] = None
        self._cal_since = 0.0
        self.cal_samples: dict[str, list[dict[str, float]]] = defaultdict(list)
        self._pending: list[Prediction] = []
        self._last: Optional[Prediction] = None
        self._raw_rec: Optional[list[tuple[np.ndarray, np.ndarray]]] = None

    # -- dashboard helpers ---------------------------------------------------------------
    @property
    def supports_stimulus(self) -> bool:
        return self.decoder.needs_stimulus

    @property
    def score_threshold(self) -> float:
        return self.decoder.threshold

    @property
    def score_scale(self) -> float:
        return getattr(self.decoder, "score_scale", 1.0)

    @property
    def smooth_windows(self) -> int:
        """Debounce for the controller's smoother: the calibrated one, else the mode's default."""
        model = getattr(self.decoder, "model", None)
        if model is not None:
            return int(model.n_smooth)
        return config.SSVEP_SMOOTH_WINDOWS if self.mode == "ssvep" else config.SMOOTH_WINDOWS

    def set_window(self, window_s: float, hop_s: Optional[float] = None) -> None:
        """Change the analysis window (a calibration chose another length). Restarts windowing."""
        self.window_s = window_s
        self.hop_s = hop_s or self.hop_s
        self.windower = Windower(window_s, self.hop_s, min_samples=int(window_s * self.fs * 0.75))

    def _adopt_window(self, cal) -> None:
        if cal is None or self.mode != "ssvep" or self._explicit_window:
            return
        hop = getattr(cal, "hop_s", None)
        if abs(cal.window_s - self.window_s) > 1e-9 or (hop and abs(hop - self.hop_s) > 1e-9):
            self.set_window(cal.window_s, hop)

    def decode_setup(self) -> str:
        """Which electrodes decode, per device, and how devices combine (saved with a calibration)."""
        parts = []
        for k, name in enumerate(self.device_names):
            dec = [self.positions[i] for i in self.decode_idx if self.device_of[i] == k]
            ref = [self.positions[i] for i in self.noise_idx if self.device_of[i] == k]
            parts.append(f"{name}:{','.join(dec)}" + (f" refs:{','.join(ref)}" if ref else ""))
        how = config.SSVEP_COMBINE if len(self.device_names) > 1 else "cca"
        return f"{how} | " + "; ".join(parts)

    def position_of(self, sdk_name: str, device: Optional[str] = None) -> str:
        """Where an SDK channel ("T4") of a device (default: the first) sits on the head."""
        return self.devices[device or self.device_names[0]]["map"][sdk_name]

    def traces(self) -> np.ndarray:
        """Last EEG_TRACE_S seconds of 1-40 Hz EEG, oldest first, shape (n, channels)."""
        return np.roll(self._trace, -self._trace_pos, axis=0)

    # -- subclass hook --------------------------------------------------------------------
    @abc.abstractmethod
    def _read_chunk(self) -> tuple[np.ndarray, np.ndarray]:
        """New samples since the last call: (timestamps[n], microvolts[n, len(sdk_channels)])."""

    # -- Source API ----------------------------------------------------------------------
    def latest(self) -> Optional[Prediction]:
        self._process()
        return self._last

    def poll(self) -> list[Prediction]:
        self._process()
        out, self._pending = self._pending, []
        return out

    # -- electrode quality ---------------------------------------------------------------
    def _update_quality(self, end: float, xd: np.ndarray, here: Optional[list[int]] = None) -> None:
        live = np.array([here is None or self.device_of[i] in here for i in range(self.channels)])
        amp = np.where(live, np.percentile(np.abs(xd), 99, axis=0), np.nan)  # no data -> not judged
        self._amp_hist.append((end, amp))
        while self._amp_hist and end - self._amp_hist[0][0] > config.EEG_QUALITY_WINDOW_S:
            self._amp_hist.popleft()
        hist = np.array([a for _, a in self._amp_hist])
        enough = np.sum(~np.isnan(hist), axis=0) >= 8   # ~2 s of windows before judging
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN column = device without data
            med = np.nanmedian(hist, axis=0)
        for i, pos in enumerate(self.positions):
            if not live[i]:
                if self.quality[pos] != "unknown":
                    self.quality[pos], self._good_since[pos] = "unknown", None
                    self.quality_note.pop(pos, None)
                    self._bad_signal.pop(pos, None)
                continue
            if not enough[i]:
                continue
            if self._raw_peak[i] > config.EEG_RAIL_UV:
                reason = "SATURATED (electrode off?)"
            elif med[i] < config.EEG_QUALITY_MIN_UV:
                reason = "FLAT (lead disconnected?)"
            elif med[i] > config.EEG_QUALITY_MAX_UV:
                reason = f"NOISY ({med[i]:.0f} µV)"
            elif self.contact.get(pos) is False:
                reason = "HIGH RESISTANCE"
            else:
                reason = None
            was = self.quality[pos]
            marginal = med[i] > config.EEG_QUALITY_WARN_UV
            if reason is not None:
                self.quality[pos], self.quality_note[pos], self._good_since[pos] = "bad", reason, None
                self._bad_signal[pos] = reason != "HIGH RESISTANCE"
            elif was == "bad":  # latched: must look good for a while before the warning clears
                if self._good_since[pos] is None:
                    self._good_since[pos] = end
                elif end - self._good_since[pos] >= config.EEG_QUALITY_RECOVER_S:
                    self.quality[pos] = "marginal" if marginal else "good"
                    self.quality_note.pop(pos, None)
                    self._bad_signal.pop(pos, None)
                    if marginal:
                        self.quality_note[pos] = f"MARGINAL ({med[i]:.0f} µV)"
            elif marginal:   # a bit noisy, still fine to decode from
                self.quality[pos], self.quality_note[pos] = "marginal", f"MARGINAL ({med[i]:.0f} µV)"
            else:
                self.quality[pos] = "good"
                self.quality_note.pop(pos, None)
            if self.quality[pos] != was and (was == "bad" or self.quality[pos] == "bad"):
                note = self.quality_note.get(pos, "ok")
                print(f"[neurorover] electrode {pos}: {self.quality[pos]} {note if self.quality[pos] == 'bad' else ''}".rstrip())

    def _unusable(self, i: int) -> bool:
        """Left out of decoding: flagged NOISY / FLAT / SATURATED (high resistance alone doesn't count)."""
        pos = self.positions[i]
        return self.quality[pos] == "bad" and self._bad_signal.get(pos, False)

    def _update_response(self, xs: np.ndarray, groups) -> None:
        targets = getattr(self.decoder, "targets", {})
        n = len(xs)
        nfft = max(n, int(4 * self.fs))       # finer frequency grid for a steadier read-out
        out = {}
        for name, cols in groups.items():
            dec = [i for i in cols if i in self.decode_idx]
            X = xs[:, dec] - xs[:, dec].mean(axis=0)
            P = np.mean(np.abs(np.fft.rfft(X * np.hanning(n)[:, None], n=nfft, axis=0)) ** 2, axis=1)
            freqs = np.fft.rfftfreq(nfft, 1.0 / self.fs)
            out[name] = {}
            for label, hz in targets.items():
                others = [h * f for f in targets.values() if f != hz for h in (1, 2)]
                out[name][label] = float(np.mean([ssvep_cal.snr_at(P, freqs, h * hz, others) for h in (1, 2)]))
        self.response = out

    def bad_electrodes(self, decode_only: bool = False) -> list[str]:
        idx = self.decode_idx if decode_only else range(len(self.positions))
        return [self.positions[i] for i in idx if self.quality[self.positions[i]] == "bad"]

    def electrode_warning(self) -> Optional[tuple[str, bool]]:
        """(message, critical) if any electrode is bad. Critical = one used for decoding."""
        bad = self.bad_electrodes()
        if not bad:
            return None
        crit = bool(self.bad_electrodes(decode_only=True))
        parts = ", ".join(f"{p} {self.quality_note.get(p, '')}".strip() for p in bad)
        return f"BAD ELECTRODE: {parts}. Reseat it" + (", then R to re-check" if hasattr(self, "measure_resistance") else ""), crit

    # -- calibration ---------------------------------------------------------------------
    def calibration_segments(self) -> list[tuple[str, float, str]]:
        """Guided calibration steps: (label, seconds, prompt)."""
        return self.decoder.calibration_segments()

    def set_calibration_label(self, label: Optional[str]) -> None:
        if label != self.calibrating:
            self.calibrating = label
            self._cal_since = self.clock()
            if label is not None:
                self._cal_log.append((label, self._cal_since))

    def calibration_timeline(self, end: Optional[float] = None) -> list[tuple[str, float, float]]:
        """The calibration steps as they happened: (label, start, end) in sample-clock time."""
        log = self._cal_log
        end = self.clock() if end is None else end
        return [(label, t, log[i + 1][1] if i + 1 < len(log) else end) for i, (label, t) in enumerate(log)]

    def fit_calibration(self, recording=None, timeline=None, progress=None):
        """Fit a calibration without changing anything live (safe to run in a worker thread).
        SSVEP with a recording: the learned decoder (pipeline/ssvep_cal.py); otherwise thresholds
        from the score samples collected during the steps."""
        if self.mode == "ssvep" and recording is not None and timeline:
            t, x = recording[0], recording[1]
            present = recording[2] if len(recording) > 2 else None
            cal = ssvep_cal.fit(t, x, present, timeline, self.layout, self.fs, self.decoder.targets,
                                progress=progress)
            cal.setup = self.decoder.setup
        else:
            samples = {k: list(v) for k, v in self.cal_samples.items()}
            cal = self.decoder.fit_samples(samples)
            cal.window_s = self.window_s
        cal.user = self.user
        return cal

    def apply_calibration(self, cal, save: bool = True):
        self.cal_samples = defaultdict(list)
        self.calibrating = None
        self._cal_log = []
        self.decoder.apply(cal)
        self.calibration = cal
        self._adopt_window(cal)
        if save:
            self.decoder.save(cal, self.models_dir)
        return cal

    def finish_calibration(self, save: bool = True):
        """Fit and apply in one go (the app does the fitting in a thread instead)."""
        rec = self.stop_recording()
        timeline = self.calibration_timeline()
        try:
            cal = self.fit_calibration(rec, timeline)
        except Exception:
            self.cancel_calibration()
            raise
        return self.apply_calibration(cal, save)

    def start_recording(self) -> None:
        """Keep every raw sample (µV, physical channel order) until stop_recording(); a calibration
        starts here, so its step log starts over too."""
        self._raw_rec = []
        self._cal_log = []

    def stop_recording(self) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """(t[n], x[n, positions], present[n, devices]) since start_recording()."""
        rec, self._raw_rec = self._raw_rec, None
        if not rec:
            return None
        return (np.concatenate([r[0] for r in rec]), np.vstack([r[1] for r in rec]),
                np.vstack([r[2] for r in rec]))

    def cancel_calibration(self) -> None:
        self.cal_samples = defaultdict(list)
        self.calibrating = None
        self._cal_log = []

    # -- pipeline ------------------------------------------------------------------------
    def _process(self) -> None:
        got = self._read_chunk()
        t, x = got[0], got[1]
        if len(t) == 0:
            return
        x = np.asarray(x, dtype=float)[:, self._perm]
        n, k_dev = len(t), len(self.device_names)
        present = (np.asarray(got[2], dtype=bool).reshape(n, k_dev) if len(got) > 2 and got[2] is not None
                   else np.ones((n, k_dev), dtype=bool))
        self._raw_peak = np.max(np.abs(x), axis=0)
        if self._raw_rec is not None:
            self._raw_rec.append((np.array(t, dtype=float), x.copy(), present.copy()))
        xs, xd = self._filter(x, present)
        self._push_trace(xd)
        muscle = (self._muscle.process(x) if self._muscle is not None
                  else np.zeros((len(x), self.channels), dtype=bool))
        both = np.hstack([xs, xd, present.astype(float), muscle.astype(float)])
        for ti, row in zip(t, both):
            for end, win in self.windower.push(float(ti), row):
                self._on_window(end, win)

    def _filter(self, x: np.ndarray, present: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run both filters; a device coming back restarts its channels' filters at its new DC level
        (otherwise the jump from the held value rings for seconds and looks like an artifact)."""
        prev = self._prev_present if self._prev_present is not None else present[0]
        before = np.vstack([prev[None, :], present[:-1]])
        starts = present & ~before
        self._prev_present = present[-1].copy()
        cuts = sorted(set(np.flatnonzero(starts.any(axis=1)).tolist()) | {0, len(x)})
        xs_parts, xd_parts = [], []
        for a, b in zip(cuts, cuts[1:]):
            for k in np.flatnonzero(starts[a]):
                cols = [i for i in range(self.channels) if self.device_of[i] == k]
                self._ssvep_filter.restart_columns(cols, x[a, cols])
                self._display_filter.restart_columns(cols, x[a, cols])
            xs_parts.append(self._ssvep_filter.process(x[a:b]))
            xd_parts.append(self._display_filter.process(x[a:b]))
        return np.vstack(xs_parts), np.vstack(xd_parts)

    def _push_trace(self, xd: np.ndarray) -> None:
        n = len(self._trace)
        for start in range(0, len(xd), n):
            block = xd[start:start + n]
            idx = (self._trace_pos + np.arange(len(block))) % n
            self._trace[idx] = block
            self._trace_pos = (self._trace_pos + len(block)) % n

    def _on_window(self, end: float, win: np.ndarray) -> None:
        c, k_dev = self.channels, len(self.device_names)
        xs, xd, pres = win[:, :c], win[:, c:2 * c], win[:, 2 * c:2 * c + k_dev]
        mus = win[:, 2 * c + k_dev:2 * c + k_dev + c] > 0.5
        here = [k for k in range(k_dev) if pres[:, k].mean() >= 0.9]   # devices with data in this window
        now_present = tuple(self.device_names[k] for k in here)
        if now_present != self.present_devices and k_dev > 1:
            share = ", ".join(f"{n} {pres[:, k].mean() * 100:.0f}%" for k, n in enumerate(self.device_names))
            print(f"[neurorover] data in window from: {' + '.join(now_present) or 'no device'} ({share})", flush=True)
        self.present_devices = now_present
        live = [i for i in range(c) if self.device_of[i] in here]
        occ_live = [i for i in self.occipital if i in live] or live or list(range(c))
        freqs, psd = spectrum(xd[:, occ_live].mean(axis=1), self.fs)
        if self.spectrum is not None and len(self.spectrum[1]) == len(psd):
            psd = 0.6 * self.spectrum[1] + 0.4 * psd  # light smoothing for display
        self.spectrum = (freqs, psd)

        # SSVEP: judge electrodes in the decoded band (slow settling drift of dry electrodes doesn't count).
        qband = xs if self.mode == "ssvep" and config.EEG_QUALITY_IN_DECODED_BAND else xd
        self._update_quality(end, qband, here)
        unusable = [i for i in range(c) if self._unusable(i)]
        presence = [1.0 if k in here else 0.0 for k in range(k_dev)]
        # Same rule as the offline calibration fit (ssvep_cal.Layout): an artifact drops that device.
        check = xs if self.mode == "ssvep" and config.SSVEP_ARTIFACT_IN_DECODED_BAND else xd
        ssvep_mode = self.mode == "ssvep"
        groups, self.device_artifact, masks = self.layout.groups(
            check, presence, unusable, muscle=mus if ssvep_mode else None,
            max_blanked=config.SSVEP_MAX_BLANKED if ssvep_mode else 0.0, fs=self.fs)
        self.blanked = max((1.0 - float(m.mean()) for m in masks.values()), default=0.0)
        self.decoding = {name: [self.positions[i] for i in cols if i in self.decode_idx]
                         for name, cols in groups.items()}
        if not groups:
            # Dropped: no prediction, never a command (artifact, or no usable electrode / no data).
            self.artifact = bool(self.device_artifact)
            self.artifact_windows += int(self.artifact)
            return
        self.artifact = False
        self.clean_windows += 1

        used = [i for cols in groups.values() for i in cols if i in self.decode_idx]
        occ = [i for i in self.occipital if i in used] or used
        if hasattr(self.decoder, "collect_subsets"):
            self.decoder.collect_subsets = self.calibrating is not None
        label, conf, self.scores = self.decoder.evaluate(xs, xd, occ, groups=groups, masks=masks)
        if self.mode == "ssvep":
            self._update_response(xs, groups)
        self.score_history.append((end, dict(self.scores), label))
        if self.calibrating is not None and end - self._cal_since >= self.decoder.cal_skip_s:
            entry = self.decoder.cal_entry(self.scores) if hasattr(self.decoder, "cal_entry") else dict(self.scores)
            self.cal_samples[self.calibrating].append(entry)
        pred = Prediction(label=label, confidence=conf, emg=None, timestamp=end, probs=dict(self.scores))
        self._last = pred
        self._pending.append(pred)
        if len(self._pending) > 200:
            del self._pending[:-200]


def resolve_devices(devices=None, sdk_channels=None, channel_map=None) -> dict[str, dict]:
    """The devices to use: given explicitly, or one device from (sdk_channels, channel_map),
    or config.EEG_DEVICES filtered by config.EEG_USE_DEVICES."""
    if devices is not None:
        if isinstance(devices, (list, tuple)):
            devices = {n: config.EEG_DEVICES[n] for n in devices}
        out = {n: dict(d) for n, d in devices.items()}
    elif sdk_channels is not None or channel_map is not None:
        sdk = tuple(sdk_channels or ("O1", "O2", "T3", "T4"))
        cmap = dict(channel_map or {c: c for c in sdk})
        missing = [c for c in sdk if c not in cmap]
        if missing:
            raise ValueError(f"channel map has no entry for SDK channel(s) {missing}")
        out = {"eeg": {"label": "EEG", "family": "any", "map": {c: cmap[c] for c in sdk}}}
    else:
        unknown = [n for n in config.EEG_USE_DEVICES if n not in config.EEG_DEVICES]
        if unknown:
            raise ValueError(f"unknown EEG device(s) {unknown}; config.EEG_DEVICES has {list(config.EEG_DEVICES)}")
        out = {n: dict(config.EEG_DEVICES[n]) for n in config.EEG_USE_DEVICES}
    if not out:
        raise ValueError("no EEG devices selected")
    return out
