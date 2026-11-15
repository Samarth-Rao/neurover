"""Alpha mode: one-bit control from occipital alpha power. Eyes closed -> F, eyes open -> S.

Score = alpha (8-12 Hz) power / total 4-30 Hz power of the occipital average. It's
a ratio, so it doesn't care about overall signal level or electrode gain. A 30 s
calibration (15 s eyes open, 15 s eyes closed) picks the threshold, erring on the
side of "eyes open never drives".
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

import config
from pipeline.eeg import band_power, spectrum

OPEN, CLOSED = "open", "closed"


def alpha_ratio(x: np.ndarray, fs: float, band=config.ALPHA_BAND, total=config.ALPHA_TOTAL_BAND) -> float:
    freqs, psd = spectrum(x, fs)
    t = band_power(freqs, psd, total)
    return band_power(freqs, psd, band) / t if t > 0 else 0.0


@dataclass
class AlphaCalibration:
    threshold: float
    open_median: Optional[float] = None
    closed_median: Optional[float] = None
    open_recall: Optional[float] = None    # share of eyes-open windows kept as S
    closed_recall: Optional[float] = None  # share of eyes-closed windows detected
    window_s: float = config.ALPHA_WINDOW_S
    user: str = "default"
    saved_at: str = ""

    def describe(self) -> str:
        return (f"alpha threshold {self.threshold:.2f} (open ~{self.open_median:.2f}, closed ~{self.closed_median:.2f}); "
                f"eyes closed detected {self.closed_recall * 100:.0f}%")


def calibrate_alpha(open_ratios: Sequence[float], closed_ratios: Sequence[float],
                    min_open_recall: float = config.ALPHA_CAL_MIN_OPEN_RECALL) -> AlphaCalibration:
    """Margin on both sides. Eyes-open alpha waxes and wanes, and eyes-closed alpha drifts between
    sessions, so when the two separate, the threshold goes midway between the top of eyes-open
    (99th percentile) and the bottom of eyes-closed (5th percentile). If they overlap, fall back to
    the threshold that keeps ``min_open_recall`` of eyes-open windows still (safety) and detects
    eyes-closed best.
    """
    o = np.asarray(open_ratios, dtype=float)
    c = np.asarray(closed_ratios, dtype=float)
    if len(o) < 5 or len(c) < 5:
        raise ValueError("not enough clean windows (need eyes open and eyes closed)")
    grid = np.linspace(min(o.min(), c.min()), max(o.max(), c.max()), 400)
    open_recall = (o[None, :] < grid[:, None]).mean(axis=1)
    closed_recall = (c[None, :] >= grid[:, None]).mean(axis=1)
    safe = open_recall >= min_open_recall
    if not safe.any():
        safe = open_recall >= open_recall.max() - 1e-9
    o_hi, c_lo = float(np.percentile(o, 99)), float(np.percentile(c, 5))
    if c_lo > o_hi:
        i = int(np.argmin(np.abs(grid - (o_hi + c_lo) / 2)))
    else:
        score = np.where(safe, closed_recall, -1.0)  # overlap: best detection among the safe thresholds
        i = int(np.flatnonzero(score >= score.max() - 1e-9)[0])
    return AlphaCalibration(threshold=round(float(grid[i]), 3), open_median=float(np.median(o)),
                            closed_median=float(np.median(c)), open_recall=float(open_recall[i]),
                            closed_recall=float(closed_recall[i]))


class AlphaDecoder:
    mode = "alpha"
    needs_stimulus = False
    score_name = "α"
    score_scale = 1.0
    classes = (CLOSED,)  # the dashboard meter shows the eyes-closed score; open is the rest state
    command_map = {CLOSED: "F", OPEN: "S"}
    class_names = {CLOSED: "Eyes closed", OPEN: "Eyes open"}
    cal_skip_s = config.ALPHA_CAL_SKIP_S

    def __init__(self, fs: float, threshold: float = config.ALPHA_THRESHOLD) -> None:
        self.fs = fs
        self.threshold = threshold

    def evaluate(self, xs: np.ndarray, xd: np.ndarray, occipital: Sequence[int], groups=None,
                 masks=None) -> tuple[str, float, dict]:
        occ = xd[:, list(occipital)].mean(axis=1) if occipital else xd.mean(axis=1)
        r = alpha_ratio(occ, self.fs)
        return (CLOSED if r >= self.threshold else OPEN), r, {CLOSED: r}

    def calibration_segments(self) -> list[tuple[str, float, str]]:
        s = config.ALPHA_CAL_SEGMENT_S
        return [(OPEN, s, "Eyes OPEN: relax and look at the rover"),
                (CLOSED, s, "CLOSE your eyes now, open them at the next beep")]

    def calibrate(self, samples: Mapping[str, Sequence[Mapping[str, float]]]) -> AlphaCalibration:
        cal = self.fit_samples(samples)
        self.apply(cal)
        return cal

    def fit_samples(self, samples: Mapping[str, Sequence[Mapping[str, float]]]) -> AlphaCalibration:
        return calibrate_alpha([s[CLOSED] for s in samples.get(OPEN, [])],
                               [s[CLOSED] for s in samples.get(CLOSED, [])])

    def apply(self, cal: AlphaCalibration) -> None:
        self.threshold = cal.threshold

    # -- persistence ------------------------------------------------------------------
    @staticmethod
    def path(user: str, models_dir: Path) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in user) or "default"
        return Path(models_dir) / f"alpha_{safe}.json"

    def save(self, cal: AlphaCalibration, models_dir: Path = config.MODELS_DIR) -> Path:
        cal.saved_at = time.strftime("%Y-%m-%d %H:%M:%S")
        p = self.path(cal.user, models_dir)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(cal.__dict__, indent=2))
        return p

    def load(self, user: str, window_s: float, models_dir: Path = config.MODELS_DIR):
        p = self.path(user, models_dir)
        if not p.exists():
            return None, None
        try:
            cal = AlphaCalibration(**json.loads(p.read_text()))
        except Exception as exc:
            return None, f"couldn't read {p.name}: {exc}"
        if abs(cal.window_s - window_s) > 1e-6:
            return None, f"ignored {p.name}: calibrated for {cal.window_s:g} s windows, now {window_s:g} s"
        self.apply(cal)
        return cal, f"loaded alpha calibration for '{user}' (threshold {cal.threshold:.2f})"
