"""Raw calibration recordings for offline method comparison (tools/ssvep_benchmark.py).

One file per completed calibration: models/<mode>_<user>_cal_<timestamp>.npz with
    t[n], x[n, channels]      EEG in µV, time.monotonic() timestamps, physical channel order
    positions, fs
    seg_labels/seg_start/seg_end   what the user was asked to look at, and when
    flip_t/flip_frame          when each flicker frame hit the screen (for phase-locked methods)
    frames_json                frames per flicker cycle for each target
    present[n, devices]        which device had data at each sample (several devices only)
    devices_json               device -> its positions, in the same order as ``present``'s columns
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

import config


def save_recording(*, mode: str, user: str, fs: float, positions: Sequence[str], t: np.ndarray, x: np.ndarray,
                   segments: Sequence[tuple[str, float, float]], flips: Sequence[tuple[float, int]],
                   frames: Mapping[str, int], refresh_hz: Optional[float], extra: Optional[dict] = None,
                   present: Optional[np.ndarray] = None, devices: Optional[Mapping[str, Sequence[str]]] = None,
                   models_dir: Path = config.MODELS_DIR) -> Path:
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in user) or "default"
    path = models_dir / f"{mode}_{safe}_cal_{time.strftime('%Y%m%d_%H%M%S')}.npz"
    flips_arr = np.asarray(flips, dtype=float).reshape(-1, 2)
    np.savez_compressed(
        path, t=np.asarray(t, dtype=float), x=np.asarray(x, dtype=float), positions=np.asarray(positions),
        fs=float(fs), seg_labels=np.asarray([s[0] for s in segments]),
        seg_start=np.asarray([s[1] for s in segments], dtype=float),
        seg_end=np.asarray([s[2] for s in segments], dtype=float),
        flip_t=flips_arr[:, 0], flip_frame=flips_arr[:, 1].astype(np.int64),
        frames_json=json.dumps(dict(frames)), refresh_hz=float(refresh_hz or config.SSVEP_REFRESH_HZ),
        meta=json.dumps({"mode": mode, "user": user, **(extra or {})}),
        present=np.ones((len(t), 1), dtype=bool) if present is None else np.asarray(present, dtype=bool),
        devices_json=json.dumps({k: list(v) for k, v in (devices or {"eeg": list(positions)}).items()}))
    return path


def load_recording(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        rec = {k: z[k] for k in z.files}
    rec["fs"] = float(rec["fs"])
    rec["refresh_hz"] = float(rec["refresh_hz"])
    rec["frames"] = json.loads(str(rec.pop("frames_json")))
    rec["meta"] = json.loads(str(rec["meta"]))
    rec["positions"] = [str(p) for p in rec["positions"]]
    # Older recordings (one device): everything present.
    rec["devices"] = json.loads(str(rec.pop("devices_json"))) if "devices_json" in rec else {"eeg": rec["positions"]}
    if "present" not in rec:
        rec["present"] = np.ones((len(rec["t"]), 1), dtype=bool)
    rec["segments"] = [(str(l), float(a), float(b)) for l, a, b in
                       zip(rec["seg_labels"], rec["seg_start"], rec["seg_end"])]
    rec["path"] = str(path)
    return rec


def list_recordings(models_dir: Path = config.MODELS_DIR, mode: str = "ssvep", user: Optional[str] = None) -> list[Path]:
    pattern = f"{mode}_{user}_cal_*.npz" if user else f"{mode}_*_cal_*.npz"
    return sorted(Path(models_dir).glob(pattern))
