"""neurover scope: every electrode's live feed, recordings, and recordings compared electrode by electrode.

    python scope.py                       # BrainBit Flex + headband together (config.EEG_USE_DEVICES)
    python scope.py --devices flex        # just the Flex
    python scope.py --mock                # synthetic EEG, no hardware
    python scope.py --view compare        # open on the comparison view
    python scope.py --report A.npz B.npz  # print the per-electrode numbers of recordings, no window
    python scope.py --trials 20           # cued sessions (T in LIVE) with 20 trials per class

--report also prints the per-trial analysis (ERD, lateralisation, classifier) of cued-trial recordings.

Click an electrode's name to say where it is; that's remembered in scope_placement.json
(--config-placement ignores it). The rover app keeps using config.EEG_DEVICES.

Recordings go to recordings/ (one .npz each; see eeg/recording.py for the format).
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import config
from eeg import cues, recording
from rawscope import analysis, placement, trials


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="neurover scope: live raw EEG, record, compare recordings")
    p.add_argument("--devices", metavar="LIST", default=",".join(config.EEG_USE_DEVICES),
                   help=f"devices to connect, from config.EEG_DEVICES ({', '.join(config.EEG_DEVICES)}); "
                        f"default {','.join(config.EEG_USE_DEVICES)}")
    p.add_argument("--mock", action="store_true", help="synthetic EEG instead of the BrainBits")
    p.add_argument("--dir", type=Path, default=recording.RECORDINGS_DIR, help="where recordings are kept")
    p.add_argument("--view", choices=("live", "compare"), default="live", help="view to open on (Tab switches)")
    p.add_argument("--config-placement", action="store_true",
                   help="ignore the electrode positions saved by clicking names; use config.EEG_DEVICES")
    p.add_argument("--report", nargs="+", metavar="NPZ", help="print per-electrode numbers for these recordings")
    p.add_argument("--reference", choices=trials.REFERENCES, default=trials.REFERENCES[0],
                   help="for --report's trial analysis: subtract each device's average (default) or as recorded")
    p.add_argument("--trials", type=int, metavar="N", default=cues.Protocol.per_class,
                   help=f"cued sessions: trials per class (default {cues.Protocol.per_class}; ↑↓ in the setup box)")
    p.add_argument("--duration", type=float, metavar="S", help="quit after S seconds (smoke tests)")
    p.add_argument("--screenshot", metavar="PNG", help="save the last frame to this file on exit")
    return p.parse_args(argv)


def report(paths, reference: str = trials.REFERENCES[0]) -> str:
    recs = [recording.load(p) for p in paths]
    stats = [analysis.analyse_recording(r) for r in recs]
    names = [r["meta"].get("name") or Path(r["path"]).stem for r in recs]
    rank = {p: i for i, p in enumerate(config.EEG_DISPLAY_ORDER)}
    positions = sorted({p for s in stats for p in s}, key=lambda p: (rank.get(p, len(rank)), p))
    width = max(10, *(len(n) for n in names)) + 2
    lines = []
    for i, (r, n) in enumerate(zip(recs, names)):
        m = r["meta"]
        devs = ", ".join(f"{d.get('label', k)} ({len(d['t'])} samples)" for k, d in r["devices"].items())
        lines.append(f"[{i + 1}] {n}: {m['duration_s']:.1f} s, {devs}{', mock' if m.get('mock') else ''}  "
                     f"{r['path']}")
    for metric in analysis.METRICS:
        lines.append("")
        lines.append(f"{metric.label} ({metric.unit})".ljust(22) + "".join(f"[{i + 1}]".rjust(width)
                                                                         for i in range(len(recs))))
        for pos in positions:
            row = f"  {pos}".ljust(22)
            for s in stats:
                v = s.get(pos, {}).get(metric.key)
                row += analysis.format_value(metric, v).rjust(width)
            lines.append(row)
    for i, (r, n) in enumerate(zip(recs, names)):
        if trials.has_trials(r["meta"]):
            lines += ["", "", f"[{i + 1}] {n}", trials.report(trials.analyse(r, reference))]
    return "\n".join(lines)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.report:
        print(report(args.report, args.reference))
        return 0
    from eeg.devices import Feeds
    from rawscope.app import ScopeApp

    names = [n.strip() for n in args.devices.split(",") if n.strip()]
    try:
        saved = {} if args.config_placement else placement.load(placement.PLACEMENT_PATH)
        feeds = Feeds(names, config.EEG_DEVICES, mock=args.mock, mock_motor=True, placement=saved)
    except ValueError as exc:
        print(f"neurover scope: {exc}", file=sys.stderr)
        return 2
    moved = {d.name for d in feeds.devices if d.positions != d.config_positions}
    if moved:
        print("[neurover scope] electrode positions from scope_placement.json: " + "; ".join(
            f"{d.label} " + ", ".join(f"{c}={p}" for c, p in zip(d.sdk, d.positions))
            for d in feeds.devices if d.name in moved), flush=True)
    ScopeApp(feeds, args.dir, view=args.view, placement_path=placement.PLACEMENT_PATH,
             protocol=cues.Protocol(per_class=max(1, args.trials))).run(
        duration=args.duration, screenshot=args.screenshot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
