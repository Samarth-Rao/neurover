"""Recordings: every device's samples exactly as they arrived, one .npz per recording.

Both apps write this format: the scope's recordings and the rover's SSVEP calibrations
(meta "kind" = "ssvep-calibration", with "segments" = what the user was asked to look at, when).

File: recordings/<YYYYmmdd_HHMMSS>[_<name>].npz with
    meta          JSON: format, name, created (unix time), start/end (monotonic), duration_s, fs, mock,
                  devices [{name, label, kind, serial, sdk[], positions[], battery,
                            resistance {position: ohms}}], events [[t_rel_s, device, text]],
                  rover calibrations: kind "ssvep-calibration", segments [[label, start, end]], flips
                  cued sessions (eeg/cues.py; scope motor trials and rover calibrations): trials [{label,
                  baseline, cue, end, practice?}]
                  with times on the samples' clock (as t<k>), and protocol {task, classes, per_class,
                  baseline_s, cue_s, pause_s, ..., seed, planned, completed, ended_early}
    t<k>, x<k>    device k (order of meta.devices): timestamps (s, monotonic clock) and
                  µV [n, channels] in that device's SDK channel order (positions[] says where each sits)

Devices are not resampled onto a shared clock; their timestamps share time.monotonic(), so they
line up to within the Bluetooth jitter.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

import config

FORMAT = "rawscope/1"
RECORDINGS_DIR = config.ROOT_DIR / "recordings"


class Recorder:
    def __init__(self, start: float, fs: float, mock: bool = False) -> None:
        self.start = start
        self.fs = fs
        self.mock = mock
        self.created = time.time()
        self._chunks: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
        self.samples: dict[str, int] = defaultdict(int)
        self.events: list[tuple[float, str, str]] = []

    def add(self, device: str, t: np.ndarray, x: np.ndarray) -> None:
        self._chunks[device].append((np.array(t, dtype=float), np.array(x, dtype=float)))
        self.samples[device] += len(t)

    def event(self, now: float, device: str, text: str) -> None:
        self.events.append((round(now - self.start, 3), device, text))

    def elapsed(self, now: float) -> float:
        return now - self.start

    def finish(self, end: float, devices: Sequence[dict]) -> dict:
        out = {}
        for meta in devices:
            chunks = self._chunks.get(meta["name"], [])
            width = len(meta["sdk"])
            t = np.concatenate([c[0] for c in chunks]) if chunks else np.zeros(0)
            x = np.vstack([c[1] for c in chunks]) if chunks else np.zeros((0, width))
            out[meta["name"]] = {**meta, "t": t, "x": x}
        return {"meta": {"format": FORMAT, "name": "", "created": self.created, "start": self.start, "end": end,
                         "duration_s": end - self.start, "fs": self.fs, "mock": self.mock,
                         "events": [list(e) for e in self.events]},
                "devices": out}


def slug(name: str) -> str:
    keep = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in name.strip())
    while "--" in keep:
        keep = keep.replace("--", "-")
    return keep.strip("-_")[:60]


def _free_path(directory: Path, stem: str, skip: Optional[Path] = None) -> Path:
    path, k = directory / f"{stem}.npz", 2
    while path.exists() and path != skip:
        path, k = directory / f"{stem}_{k}.npz", k + 1
    return path


def save(rec: dict, directory: Path = RECORDINGS_DIR, name: str = "", replace: Optional[Path] = None) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    meta = dict(rec["meta"])
    meta["name"] = name.strip()
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(meta["created"]))
    s = slug(name)
    path = _free_path(directory, f"{stamp}_{s}" if s else stamp, skip=replace)
    arrays, devs = {}, []
    for k, d in enumerate(rec["devices"].values()):
        arrays[f"t{k}"] = np.asarray(d["t"], dtype=float)
        arrays[f"x{k}"] = np.asarray(d["x"], dtype=np.float32)
        devs.append({key: v for key, v in d.items() if key not in ("t", "x")})
    meta["devices"] = devs
    tmp = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(tmp, meta=np.asarray(json.dumps(meta)), **arrays)
    tmp.replace(path)
    return path


def load(path) -> dict:
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        devices = {}
        for k, d in enumerate(meta.pop("devices")):
            devices[d["name"]] = {**d, "t": z[f"t{k}"].astype(float), "x": z[f"x{k}"].astype(float)}
    return {"meta": meta, "devices": devices, "path": path}


def info(path) -> dict:
    """Metadata only (cheap: the sample arrays aren't decompressed)."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
    meta["path"] = path
    meta["mtime"] = path.stat().st_mtime
    return meta


def rename(path, name: str) -> Path:
    """Give a recording a (new) name: the name is in its metadata and its file name."""
    path = Path(path)
    new = save(load(path), path.parent, name, replace=path)
    if new != path:
        path.unlink()
    return new


def relabel(path, device: str, sdk: str, position: str) -> Path:
    """Fix where one electrode of a saved recording was (same rules as moving it live: swaps within the
    device, refused if another device has that spot). Its resistance value moves with it."""
    from .devices import reassign

    path = Path(path)
    rec = load(path)
    devs = rec["devices"]
    d = devs[device]
    old = list(d["positions"])
    by_sdk = {c: d.get("resistance", {}).get(p) for c, p in zip(d["sdk"], old)}
    new = reassign({k: v["positions"] for k, v in devs.items()}, device, d["sdk"].index(sdk), position,
                   {k: v.get("label", k) for k, v in devs.items()})
    d["positions"] = new[device]
    d["resistance"] = {p: by_sdk[c] for c, p in zip(d["sdk"], d["positions"]) if by_sdk[c] is not None}
    return save(rec, path.parent, rec["meta"].get("name", ""), replace=path)


def list_recordings(directory: Path = RECORDINGS_DIR) -> list[dict]:
    """Every readable recording in ``directory``, newest first."""
    out = []
    for p in Path(directory).glob("*.npz"):
        if p.name.endswith(".partial.npz"):
            continue
        try:
            meta = info(p)
        except Exception:
            continue
        if meta.get("format") == FORMAT:
            out.append(meta)
    return sorted(out, key=lambda m: m.get("created", 0), reverse=True)


def nodes(rec: dict) -> list[dict]:
    """Every electrode in a recording: position, device, SDK channel, its samples and resistance."""
    out = []
    for d in rec["devices"].values():
        for i, (sdk, pos) in enumerate(zip(d["sdk"], d["positions"])):
            out.append({"position": pos, "device": d["name"], "label": d.get("label", d["name"]), "sdk": sdk,
                        "t": d["t"], "x": d["x"][:, i] if d["x"].ndim == 2 and len(d["x"]) else np.zeros(0),
                        "resistance": d.get("resistance", {}).get(pos)})
    return out
