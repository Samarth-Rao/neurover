"""Replay SSVEP calibration recordings through the rover's decoder, offline.

    python tools/ssvep_eval.py recordings/<calibration>.npz        # a calibration made by the rover (C)
    python tools/ssvep_eval.py models/ssvep_<user>_cal_<time>.npz  # older calibration files work too
    python tools/ssvep_eval.py --together A.npz B.npz C.npz      # several fitted as one, like the rover does

For each recording it prints how far the boxes stood out while the driver looked at them vs at the
centre, and what the calibration fit would choose (or why it refuses). Use it to compare electrode
setups: record a calibration with each, then compare the numbers.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eeg import recording, ssvep  # noqa: E402


def load(path) -> dict:
    """A recording in eeg/recording.py format; older calibration files (one shared timeline) are converted."""
    with np.load(path, allow_pickle=False) as z:
        if "meta" in z.files and json.loads(str(z["meta"])).get("format") == recording.FORMAT:
            return recording.load(path)
        old = {k: z[k] for k in z.files}
    t, x = old["t"], old["x"]
    positions = [str(p) for p in old["positions"]]
    devices = json.loads(str(old["devices_json"])) if "devices_json" in old else {"eeg": positions}
    present = old.get("present", np.ones((len(t), 1), dtype=bool)).astype(bool)
    if present.shape[1] != len(devices):
        present = np.ones((len(t), len(devices)), dtype=bool)
    devs = {}
    for k, (name, ps) in enumerate(devices.items()):
        cols = [positions.index(p) for p in ps]
        keep = present[:, k]
        devs[name] = {"name": name, "label": name, "sdk": list(ps), "positions": list(ps), "t": t[keep],
                      "x": x[keep][:, cols]}
    segments = [(str(l), float(a), float(b)) for l, a, b in zip(old["seg_labels"], old["seg_start"], old["seg_end"])]
    meta = {"format": recording.FORMAT, "fs": float(old["fs"]), "start": float(t[0]), "end": float(t[-1]),
            "duration_s": float(t[-1] - t[0]), "segments": segments, "kind": "ssvep-calibration (old format)"}
    return {"meta": meta, "devices": devs, "path": Path(path)}


def separation(rec: dict, window_s: float) -> dict:
    """Pooled z in settled windows (> 1 s into a trial): for the box looked at, the other boxes, and the
    best box while looking at the centre."""
    labels = list(ssvep.target_frequencies())
    trials = ssvep.calibration_trials(rec["meta"])
    t, z = ssvep.replay(rec, window_s)
    looked, others, rest_best, picks = [], [], [], []
    for tr in trials:
        sel = (t >= tr.onset + max(1.0, window_s)) & (t <= tr.end) & ~np.isnan(z).any(axis=1)
        if tr.label == ssvep.REST:
            rest_best += z[sel].max(axis=1).tolist()
        elif tr.label in labels:
            i = labels.index(tr.label)
            looked += z[sel, i].tolist()
            others += np.delete(z[sel], i, axis=1).ravel().tolist()
            picks += (np.argmax(z[sel], axis=1) == i).tolist()
    q = lambda v, p: float(np.percentile(v, p)) if v else float("nan")  # noqa: E731
    return {"looked_median": q(looked, 50), "others_median": q(others, 50), "centre_best_median": q(rest_best, 50),
            "centre_best_p95": q(rest_best, 95), "pick": float(np.mean(picks)) if picks else float("nan"),
            "windows": len(looked) + len(rest_best)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("paths", nargs="+")
    p.add_argument("--no-fit", action="store_true", help="only the separation numbers")
    p.add_argument("--together", action="store_true",
                   help="also fit all the given calibrations together, as the rover does with repeated ones")
    args = p.parse_args(argv)
    for path in args.paths:
        rec = load(path)
        devs = ", ".join(f"{d.get('label', n)} {d['positions']}" for n, d in rec["devices"].items())
        print(f"\n{Path(path).name}: {rec['meta']['duration_s']:.0f} s, {devs}")
        for w in (1.0, 2.0, 3.0):
            s = separation(rec, w)
            print(f"  {w:g} s windows: box looked at z {s['looked_median']:+.2f} (other boxes {s['others_median']:+.2f}),"
                  f" centre's best box z {s['centre_best_median']:+.2f} (95th pct {s['centre_best_p95']:+.2f}),"
                  f" right box picked {s['pick'] * 100:.0f}% (chance 25%)")
        if not args.no_fit:
            try:
                st = ssvep.fit(rec)
                print(f"  calibration would choose: z >= {st.threshold}, margin {st.margin}, debounce {st.n_smooth}; "
                      f"{st.describe()}")
            except ValueError as exc:
                print(f"  calibration refuses: {exc}")
    if args.together and len(args.paths) > 1:
        try:
            st = ssvep.fit([load(path) for path in args.paths])
            print(f"\nall {len(args.paths)} together: z >= {st.threshold}, margin {st.margin}, debounce {st.n_smooth}; "
                  f"{st.describe()}")
        except ValueError as exc:
            print(f"\nall {len(args.paths)} together: calibration refuses: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
