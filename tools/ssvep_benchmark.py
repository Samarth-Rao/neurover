"""Compare SSVEP electrode setups and methods on saved calibration recordings.

    python tools/ssvep_benchmark.py a.npz b.npz           # which electrodes / devices decode best
    python tools/ssvep_benchmark.py --methods a.npz b.npz # which scoring method / fusion decodes best
    python tools/ssvep_benchmark.py                       # all models/ssvep_*_cal_*.npz

Electrode setups (default report), built from the devices in the recordings, e.g. Flex + headband:
    flex            the Flex's decoding electrodes (PO3, POz, PO4)        one CCA
    flex +Pz        ... plus its other electrode(s)
    headband        O1, O2
    headband +T3,T4 ... plus the temporal electrodes as noise references
    joint           one CCA over every decoding electrode of both devices
    fused           CCA per device, correlations averaged (the live default with 2 devices)
    fused +others   the same, each device with its extra electrodes

Cross-session: everything is trained/calibrated on one recording and tested on another (both
directions), because that's what live use looks like. With a single recording, it trains on the
first half of every segment and tests on the second half (and vice versa): optimistic, but useful.
For each window length it reports:

    pick%   4-way accuracy on settled target windows (which box, ignoring rest) - chance 25%
    rest%   centre-gaze windows the decision keeps as rest (after the safety-first threshold)
    det%    target windows detected as the right box
    t_cmd   median seconds from looking at a box to the rover getting that command (smoother N=4)
    miss    target segments where the right command never came
    drift%  share of centre-gaze time the rover was driving anyway (false commands)
Methods ending in "+ew" smooth scores over ~0.5 s (exponential average) and use a 2-window debounce.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from pipeline import ssvep, ssvep_data  # noqa: E402
from pipeline import ssvep_methods as sm  # noqa: E402
from pipeline.smoother import Smoother  # noqa: E402

WINDOWS = (1.0, 1.5, 2.0, 3.0)
FUSED = ("avg", "weighted", "stacked")
EW_TAU_S = 0.5


def oof_trca(p, wins, n):
    """Out-of-fold TRCA scores for a training recording: first vs second half of each segment."""
    halves = {}
    for i, w in enumerate(wins):
        a, b = p.segments[w.seg][1], p.segments[w.seg][2]
        halves.setdefault(int(w.t0 >= (a + b) / 2), []).append(i)
    out = np.zeros((len(wins), len(p.targets)))
    for k in (0, 1):
        fit_on = [wins[i] for i in halves.get(1 - k, [])]
        model = sm.Trca().fit(p, fit_on, n)
        for i in halves.get(k, []):
            out[i] = model.score(p, wins[i], n)
    return out


def ewma(scores: np.ndarray, wins, tau_s: float) -> np.ndarray:
    out = np.zeros_like(scores)
    prev, prev_t = None, None
    for i, w in enumerate(wins):
        if prev is None or (prev_t is not None and w.t_end - prev_t > 1.0):
            prev = scores[i]
        else:
            a = 1.0 - math.exp(-(w.t_end - prev_t) / tau_s)
            prev = prev + a * (scores[i] - prev)
        prev_t = w.t_end
        out[i] = prev
    return out


def evaluate(train_scores, train_wins, test_scores, test_wins, p_test, labels, n_smooth):
    def samples(scores, wins):
        d = {}
        for s, w in zip(scores, wins):
            if w.clean and w.settled:
                d.setdefault(w.label, []).append(dict(zip(labels, s)))
        return d

    cal = ssvep.calibrate(samples(train_scores, train_wins), labels)
    dec = ssvep.SsvepDecoder(p_test.fs, p_test.targets, threshold=cal.threshold, margin=cal.margin)
    y = np.array([w.label for w in test_wins])
    ok = np.array([w.clean and w.settled for w in test_wins])
    tgt = ok & (y != sm.REST)
    pick = float(np.mean(np.array(labels)[np.argmax(test_scores[tgt], axis=1)] == y[tgt]))
    decided = np.array([dec.decide(dict(zip(labels, s)))[0] for s in test_scores])
    rest_kept = float(np.mean(decided[ok & (y == sm.REST)] == sm.REST)) if np.any(ok & (y == sm.REST)) else float("nan")
    det = float(np.mean(decided[tgt] == y[tgt]))

    # Live simulation through the smoother, in time order (artifact windows produce nothing).
    smoother = Smoother(n_windows=n_smooth, threshold=0.0)
    cmd_at = []
    for w, lab in zip(test_wins, decided):
        if w.clean:
            smoother.update(config.SSVEP_COMMANDS[lab], 1.0, w.t_end)
        cmd_at.append(smoother.output(w.t_end))
    t_cmd, misses, drift_n, drift_tot = [], 0, 0, 0
    for si, (label, a, b) in enumerate(p_test.segments):
        idx = [i for i, w in enumerate(test_wins) if w.seg == si]
        if label == sm.REST:
            settled = [i for i in idx if test_wins[i].t_end >= a + 2.5]
            drift_n += sum(cmd_at[i] != "S" for i in settled)
            drift_tot += len(settled)
            continue
        want = config.SSVEP_COMMANDS[label]
        hit = next((test_wins[i].t_end - a for i in idx if cmd_at[i] == want), None)
        if hit is None:
            misses += 1
        else:
            t_cmd.append(hit)
    return dict(pick=pick, rest=rest_kept, det=det, t_cmd=float(np.median(t_cmd)) if t_cmd else float("nan"),
                miss=misses, drift=drift_n / drift_tot if drift_tot else float("nan"))


def run_pair(rec_train, rec_test, window_s):
    pa, pb = sm.prepare(rec_train), sm.prepare(rec_test)
    labels = list(pa.targets)
    n = int(round(window_s * pa.fs))
    wa, wb = sm.windows(pa, window_s), sm.windows(pb, window_s)
    trca_full = sm.Trca().fit(pa, wa, n)
    sa = sm.score_all(pa, wa, window_s, trca=None)
    sa.by_method["trca"] = oof_trca(pa, wa, n)          # training scores without leakage
    sb = sm.score_all(pb, wb, window_s, trca=trca_full)
    fusion = sm.Fusion(sm.BASE_METHODS).fit(sa, labels=labels)
    results = {}
    for m in sm.BASE_METHODS:
        results[m] = (sa.by_method[m], sb.by_method[m])
    for how in FUSED:
        results[how] = (fusion.combine(sa, how, labels), fusion.combine(sb, how, labels))
    out = {}
    for name, (tr, te) in results.items():
        out[name] = evaluate(tr, wa, te, wb, pb, labels, config.SMOOTH_WINDOWS)
        out[name + "+ew"] = evaluate(ewma(tr, wa, EW_TAU_S), wa, ewma(te, wb, EW_TAU_S), wb, pb, labels, 2)
    return out


# -- electrode setups ------------------------------------------------------------------------------
def montages(rec) -> dict[str, list[list[str]]]:
    """Setup name -> groups of positions (one CCA per group, correlations averaged across groups)."""
    devs = rec["devices"]
    dec = {d: [p for p in ps if p in config.EEG_DECODE_POSITIONS] for d, ps in devs.items()}
    extra = {d: [p for p in ps if p not in config.EEG_DECODE_POSITIONS] for d, ps in devs.items()}
    out: dict[str, list[list[str]]] = {}
    for d in devs:
        if dec[d]:
            out[d] = [dec[d]]
            if extra[d]:
                out[f"{d} +{','.join(extra[d])}"] = [dec[d] + extra[d]]
    both = [d for d in devs if dec[d]]
    if len(both) > 1:
        out["joint"] = [[p for d in both for p in dec[d]]]
        out["fused"] = [dec[d] for d in both]
        if any(extra[d] for d in both):
            out["fused +others"] = [dec[d] + extra[d] for d in both]
    return out


def score_groups(rec, groups, window_s, method):
    """Windows (clean only if every group is clean) and scores averaged over the groups."""
    n = int(round(window_s * rec["fs"]))
    fn = {"cca": sm.score_cca, "fbcca": sm.score_fbcca}[method]
    preps = [sm.prepare(rec, g) for g in groups]
    wins_per = [sm.windows(p, window_s) for p in preps]
    wins = [sm.Window(**{**w.__dict__, "clean": all(ws[i].clean for ws in wins_per)})
            for i, w in enumerate(wins_per[0])]
    scores = np.mean([[fn(p, w, n) for w in ws] for p, ws in zip(preps, wins_per)], axis=0)
    return preps[0], wins, scores


def halves(rec):
    """Two pseudo-sessions from one recording: first and second half of every segment."""
    a, b = dict(rec), dict(rec)
    a["segments"] = [(l, s, (s + e) / 2) for l, s, e in rec["segments"]]
    b["segments"] = [(l, (s + e) / 2, e) for l, s, e in rec["segments"]]
    a["path"], b["path"] = rec["path"] + " (1st halves)", rec["path"] + " (2nd halves)"
    return a, b


def compare_montages(recs, pairs):
    setups = montages(recs[0])
    for r in recs[1:]:
        setups = {k: v for k, v in setups.items() if k in montages(r)}
    if not setups:
        print("these recordings have no electrode setup in common (different devices or positions); "
              "pass recordings made with the same setup")
        return
    for window_s in WINDOWS:
        print(f"\nwindow {window_s:g} s (mean of {len(pairs)} train/test directions)")
        print(f"  {'electrodes':26s} {'method':6s} {'pick%':>6s} {'rest%':>6s} {'det%':>6s} {'t_cmd':>6s} "
              f"{'miss':>5s} {'drift%':>7s}")
        for name, groups in setups.items():
            for method in ("cca", "fbcca"):
                rs = []
                for a, b in pairs:
                    pa, wa, sa = score_groups(recs[a], groups, window_s, method)
                    pb, wb, sb = score_groups(recs[b], groups, window_s, method)
                    try:
                        rs.append(evaluate(sa, wa, sb, wb, pb, list(pa.targets), config.SMOOTH_WINDOWS))
                    except ValueError:   # e.g. a bad electrode made every centre window an artifact
                        pass
                label = name if method == "cca" else ""
                row = _row(rs) if rs else "   too few clean windows (a bad electrode in this set?)"
                print(f"  {label:26s} {method:6s} " + row)


def _row(rs) -> str:
    def mean(k):
        vals = [r[k] for r in rs if not (isinstance(r[k], float) and math.isnan(r[k]))]
        return float(np.mean(vals)) if vals else float("nan")
    return (f"{mean('pick') * 100:6.0f} {mean('rest') * 100:6.0f} {mean('det') * 100:6.0f} "
            f"{mean('t_cmd'):6.1f} {sum(r['miss'] for r in rs):5d} {mean('drift') * 100:7.1f}")


def main(paths, methods: bool = False):
    recs = [ssvep_data.load_recording(p) for p in paths]
    if not recs:
        print("no calibration recordings found (press C in the app; each run saves one)")
        return 1
    print(f"recordings: {', '.join(Path(r['path']).name for r in recs)}")
    devs = recs[0]["devices"]
    print("devices: " + "; ".join(f"{d}: {', '.join(ps)}" for d, ps in devs.items()))
    if len(recs) == 1:
        print("one recording: training on the first half of each segment, testing on the second (and back). "
              "Calibrate again for a fairer cross-session test.")
        recs = list(halves(recs[0]))
    pairs = [(a, b) for a in range(len(recs)) for b in range(len(recs)) if a != b]
    if not methods:
        compare_montages(recs, pairs)
        return 0
    print(f"decode electrodes: {', '.join(sm.prepare(recs[0]).positions)}")
    for window_s in WINDOWS:
        agg = {}
        for a, b in pairs:
            for name, r in run_pair(recs[a], recs[b], window_s).items():
                agg.setdefault(name, []).append(r)
        print(f"\nwindow {window_s:g} s (train on one session, test on the other; mean of {len(pairs)} directions)")
        print(f"  {'method':14s} {'pick%':>6s} {'rest%':>6s} {'det%':>6s} {'t_cmd':>6s} {'miss':>5s} {'drift%':>7s}")
        for name, rs in agg.items():
            print(f"  {name:14s} " + _row(rs))
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    want_methods = "--methods" in args
    files = [a for a in args if a != "--methods"]
    sys.exit(main(files or [str(p) for p in ssvep_data.list_recordings(config.MODELS_DIR, "ssvep")], want_methods))
