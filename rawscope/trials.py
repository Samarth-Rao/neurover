"""Cued-trial analysis: does moving or imagining a hand show as mu / beta desynchronisation (ERD)?

Each trial is compared with its own baseline (the fixation cross just before its cue), so slow drift
(electrodes settling over the session) cancels out:

    ERD (dB) = 10·log10(band power 0.5-4 s after the cue / band power in the 2 s before it)

Negative = desynchronised: less rhythm while the hand moves or is imagined. Expected: right hand at
C3, left hand at C4 (the opposite hemisphere).

    per electrode, per class   mean ERD over trials, SE, t-test against 0; left vs right (Welch)
    time course                band power vs that trial's baseline, -2..+5 s around the cue, per class
    lateralisation             C4 - C3 ERD per trial (left hand → negative, right hand → positive)
    classifier                 per device: shrinkage LDA on each trial's log band power 0.5-4 s after
                               the cue (leads × bands), stratified K-fold over trials, against N_PERM
                               label shuffles (chance level and a p-value). Not baseline-corrected:
                               a 2 s baseline adds as much noise as it removes, and with the classes
                               interleaved, drift can't line up with a class.

Reference: "device" (the default) subtracts each device's mean across its leads, sample by sample.
Within a device the leads share its reference and movement noise (60-97% of the power in real
recordings) and each device has its own reference, so each is referenced separately. "raw" = as
recorded. Artifact rejection uses the chosen reference: a trial is dropped on a device if any of its
leads goes over EEG_ARTIFACT_UV (1-40 Hz) from -2 to +4 s, or if the device has a gap there.
"""
from __future__ import annotations

import math
import warnings
from typing import Optional, Sequence

import numpy as np
from scipy import signal, stats

import config

from . import analysis

BANDS: tuple[tuple[str, tuple[float, float]], ...] = (("mu", (8.0, 13.0)), ("beta", (13.0, 30.0)))
BASELINE_S = (-2.0, 0.0)   # windows (s from the cue) for the default protocol; see windows()
TASK_S = (0.5, 4.0)
COURSE_S = (-2.0, 5.0)
COURSE_STEP_S = 0.1
SMOOTH_S = 0.5             # the time course is band power averaged over this long
EDGE_S = 0.5               # filter edges: an epoch must be this far inside continuous data
ARTIFACT_UV = config.EEG_ARTIFACT_UV
N_PERM = 200
FOLDS = 5
MIN_PER_CLASS = 4          # for the classifier
PAIR = ("left", "right")   # what the classifier separates (if both are there; else every class)
MOTOR = ("C3", "Cz", "C4")
REFERENCES = ("device", "raw")
REFERENCE_LABEL = {"device": "device average", "raw": "as recorded"}

_SOS: dict[tuple, np.ndarray] = {}


def has_trials(meta: dict) -> bool:
    """A cued motor session (the rover's SSVEP calibrations have cued trials too, but not for this)."""
    if meta.get("kind") == "ssvep-calibration":
        return False
    return any(not t.get("practice") and "cue" in t for t in meta.get("trials") or [])


def windows(meta: dict) -> dict:
    """Baseline, task and time-course windows (s from the cue) for a recording, from its protocol's
    fixation and cue lengths, so the task window never runs into the pause. The default protocol
    gives BASELINE_S, TASK_S and COURSE_S."""
    proto = meta.get("protocol") or {}
    base = float(proto.get("baseline_s", -BASELINE_S[0]))
    cue = float(proto.get("cue_s", TASK_S[1]))
    return {"baseline": (-base, 0.0), "task": (min(TASK_S[0], cue / 2), cue),
            "course": (-base, cue + COURSE_S[1] - TASK_S[1])}


def _band_sos(band: tuple[float, float], fs: float) -> np.ndarray:
    key = (band, fs)
    if key not in _SOS:
        _SOS[key] = signal.butter(4, band, btype="bandpass", fs=fs, output="sos")
    return _SOS[key]


def _filter(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    y = signal.detrend(x, axis=0, type="linear")
    return signal.sosfiltfilt(sos, y, axis=0)


def reference(x: np.ndarray, kind: str) -> np.ndarray:
    """``x`` [n, leads] of one device, re-referenced."""
    if kind == "device" and x.shape[1] >= 2:
        return x - x.mean(axis=1, keepdims=True)
    return x


def _moving_mean(p: np.ndarray, w: int) -> np.ndarray:
    w = max(1, w)
    cs = np.vstack([np.zeros((1, p.shape[1])), np.cumsum(p, axis=0)])
    i = np.arange(len(p))
    lo = np.clip(i - w // 2, 0, len(p))
    hi = np.clip(i - w // 2 + w, 0, len(p))
    return (cs[hi] - cs[lo]) / np.maximum(1, hi - lo)[:, None]


def _db(ratio):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 10.0 * np.log10(ratio)
    return np.where(np.isfinite(out), out, np.nan)


def epochs(d: dict, cues: np.ndarray, fs: float, ref: str, times: np.ndarray, win: Optional[dict] = None) -> dict:
    """One device's trials: "erd" [trial, lead, band] dB, "power" [trial, lead, band] (task-window
    band power, dB re 1 µV²), "course" [trial, lead, band, time] dB and "status" per trial (ok /
    artifact / missing). Dropped trials are NaN."""
    win = win or windows({})
    t = np.asarray(d["t"], dtype=float)
    x = np.asarray(d["x"], dtype=float).reshape(len(t), -1) if len(t) else np.zeros((0, len(d["sdk"])))
    n, c, nb = len(cues), x.shape[1], len(BANDS)
    erd = np.full((n, c, nb), np.nan)
    pw = np.full((n, c, nb), np.nan)
    course = np.full((n, c, nb, len(times)), np.nan)
    status = ["missing"] * n
    for a, b in analysis.runs(t):
        tt = t[a:b]
        inside = (cues + win["baseline"][0] - EDGE_S >= tt[0]) & (cues + win["task"][1] + EDGE_S <= tt[-1])
        if b - a < 2 * fs or not inside.any():
            continue
        xr = reference(x[a:b], ref)
        y40 = np.abs(_filter(xr, analysis._sos(fs)))
        power = [_filter(xr, _band_sos(band, fs)) ** 2 for _, band in BANDS]
        smooth = [_moving_mean(p, int(SMOOTH_S * fs)) for p in power]
        for k in np.flatnonzero(inside):
            cue = cues[k]
            i0, i1, w0, w1 = np.searchsorted(tt, cue + np.array([*win["baseline"], *win["task"]]))
            if float(np.max(y40[i0:w1])) > ARTIFACT_UV:
                status[k] = "artifact"
                continue
            status[k] = "ok"
            at = cue + times
            ti = np.clip(np.searchsorted(tt, at), 0, len(tt) - 1)
            ok = (at - SMOOTH_S / 2 >= tt[0]) & (at + SMOOTH_S / 2 <= tt[-1])
            for j in range(nb):
                base = power[j][i0:i1].mean(axis=0)
                pw[k, :, j] = _db(power[j][w0:w1].mean(axis=0))
                erd[k, :, j] = pw[k, :, j] - _db(base)
                course[k, :, j][:, ok] = _db(smooth[j][ti[ok]] / base).T
    return {"erd": erd, "power": pw, "course": course, "status": status}


def stat(v) -> dict:
    """Mean, SE, n and a one-sample t-test against 0."""
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    out = {"mean": math.nan, "se": math.nan, "n": len(v), "t": math.nan, "p": math.nan}
    if len(v):
        out["mean"] = float(v.mean())
    if len(v) >= 2:
        sd = float(v.std(ddof=1))
        out["se"] = sd / math.sqrt(len(v))
        if sd > 0:
            r = stats.ttest_1samp(v, 0.0)
            out["t"], out["p"] = float(r.statistic), float(r.pvalue)
    return out


def compare(a, b) -> dict:
    """Mean of ``a`` minus mean of ``b``, Welch's t-test."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    out = {"diff": math.nan, "t": math.nan, "p": math.nan, "n": (len(a), len(b))}
    if len(a) and len(b):
        out["diff"] = float(a.mean() - b.mean())
    if len(a) >= 2 and len(b) >= 2 and (a.std() > 0 or b.std() > 0):
        r = stats.ttest_ind(a, b, equal_var=False)
        out["t"], out["p"] = float(r.statistic), float(r.pvalue)
    return out


def _mean_se(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and SE over axis 0, ignoring NaN (NaN where nothing is left)."""
    ok = np.isfinite(v)
    n = ok.sum(axis=0)
    s = np.where(ok, v, 0.0)
    mean = np.where(n > 0, s.sum(axis=0) / np.maximum(n, 1), np.nan)
    var = np.where(ok, (v - mean) ** 2, 0.0).sum(axis=0) / np.maximum(n - 1, 1)
    se = np.where(n > 1, np.sqrt(var / np.maximum(n, 1)), np.nan)
    return mean, se


def classify(X: np.ndarray, y: Sequence[str], n_perm: int = N_PERM, seed: int = 0) -> dict:
    """Cross-validated shrinkage-LDA accuracy, and the same with shuffled labels (chance, p-value)."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(y)
    labels, counts = np.unique(y, return_counts=True)
    out = {"n": len(y), "features": X.shape[1] if X.ndim == 2 else 0, "classes": labels.tolist(),
           "acc": math.nan, "chance": math.nan, "p": math.nan, "null": np.zeros(0), "folds": 0, "error": ""}
    if len(labels) < 2 or counts.min() < MIN_PER_CLASS:
        out["error"] = f"needs {MIN_PER_CLASS}+ clean trials of each class"
        return out
    folds = int(min(FOLDS, counts.min()))
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)

    def accuracy(labels_: np.ndarray) -> float:
        right = 0
        for train, test in cv.split(X, labels_):
            lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X[train], labels_[train])
            right += int(np.sum(lda.predict(X[test]) == labels_[test]))
        return right / len(labels_)

    rng = np.random.default_rng(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        acc = accuracy(y)
        null = np.array([accuracy(rng.permutation(y)) for _ in range(n_perm)])
    out.update(acc=acc, null=null, folds=folds, chance=float(null.mean()) if n_perm else math.nan,
               p=float((1 + np.sum(null >= acc - 1e-12)) / (1 + n_perm)) if n_perm else math.nan)
    return out


def _class_order(meta: dict, trials: list[dict]) -> list[str]:
    proto = meta.get("protocol") or {}
    seen = [c for c in proto.get("classes", []) if any(t["label"] == c for t in trials)]
    for t in trials:
        if t["label"] not in seen:
            seen.append(t["label"])
    return seen


def analyse(rec: dict, ref: str = "device", *, n_perm: int = N_PERM, seed: int = 0) -> dict:
    """Everything the trial view and the report show, for one recording and one reference."""
    if ref not in REFERENCES:
        raise ValueError(f"reference must be one of {REFERENCES}")
    meta = rec["meta"]
    fs = float(meta["fs"])
    trials = [t for t in meta.get("trials") or [] if not t.get("practice") and "cue" in t]
    classes = _class_order(meta, trials)
    labels = np.array([t["label"] for t in trials])
    cues = np.array([float(t["cue"]) for t in trials])
    win = windows(meta)
    times = np.round(np.arange(win["course"][0], win["course"][1] + 1e-9, COURSE_STEP_S), 3)
    proto = meta.get("protocol") or {}
    out = {"reference": ref, "task": proto.get("task", ""), "classes": classes, "n_trials": len(trials),
           "windows": win,
           "by_class": {c: int(np.sum(labels == c)) for c in classes}, "times": times, "devices": {},
           "electrodes": [], "lateralisation": None, "classifier": {}, "n_perm": n_perm}
    pair = [c for c in PAIR if c in classes]
    pair = pair if len(pair) == 2 else classes
    rank = {p: i for i, p in enumerate(config.EEG_DISPLAY_ORDER)}
    per_lead = {}           # position -> per-trial ERD [trial, band]
    for d in rec["devices"].values():
        ep = epochs(d, cues, fs, ref, times, win)
        st = np.array(ep["status"]) if trials else np.zeros(0, dtype=str)
        keep = st == "ok"
        out["devices"][d["name"]] = {
            "label": d.get("label", d["name"]), "total": len(trials), "kept": int(keep.sum()),
            "artifact": int(np.sum(st == "artifact")), "missing": int(np.sum(st == "missing")),
            "by_class": {c: int(np.sum(keep & (labels == c))) for c in classes}}
        for col, (sdk, pos) in enumerate(zip(d["sdk"], d["positions"])):
            e = {"position": pos, "device": d["name"], "label": d.get("label", d["name"]), "sdk": sdk,
                 "kept": int(keep.sum()), "erd": {}, "lr": {}, "course": {}}
            for j, (band, _) in enumerate(BANDS):
                v = ep["erd"][:, col, j]
                e["erd"][band] = {c: stat(v[labels == c]) for c in classes}
                e["lr"][band] = compare(v[labels == "left"], v[labels == "right"]) if "left" in classes \
                    and "right" in classes else None
                e["course"][band] = {c: _mean_se(ep["course"][labels == c, col, j]) for c in classes}
            per_lead[pos] = ep["erd"][:, col, :]
            out["electrodes"].append(e)
        sel = keep & np.isin(labels, pair)
        X = ep["power"][sel].reshape(int(sel.sum()), ep["power"].shape[1] * len(BANDS))
        good = np.all(np.isfinite(X), axis=1) if len(X) else np.zeros(0, dtype=bool)
        out["classifier"][d["name"]] = classify(X[good], labels[sel][good], n_perm, seed)
    # C3, Cz, C4 first; then device by device, in the dashboard's order
    devices = list(rec["devices"])
    out["electrodes"].sort(key=lambda e: (0, MOTOR.index(e["position"])) if e["position"] in MOTOR else
                           (1, devices.index(e["device"]), rank.get(e["position"], len(rank)), e["sdk"]))
    if "C3" in per_lead and "C4" in per_lead:
        li = per_lead["C4"] - per_lead["C3"]
        out["lateralisation"] = {
            band: {"by_class": {c: stat(li[labels == c, j]) for c in classes},
                   "lr": compare(li[labels == "left", j], li[labels == "right", j])
                   if "left" in classes and "right" in classes else None}
            for j, (band, _) in enumerate(BANDS)}
    return out


# -- text -------------------------------------------------------------------------------------
def _stars(p: float) -> str:
    return "**" if p < 0.01 else "*" if p < 0.05 else ""


def _p(p: float) -> str:
    if not math.isfinite(p):
        return "–"
    return f"{p:.3g}" if p >= 0.001 else f"{p:.0e}"


def _cell(s: dict) -> str:
    if not math.isfinite(s["mean"]):
        return "–"
    se = f" ± {s['se']:.2f}" if math.isfinite(s["se"]) else ""
    return f"{s['mean']:+.2f}{se} ({s['n']}){_stars(s['p'])}"


def band_label(band: str) -> str:
    lo, hi = dict(BANDS)[band]
    return f"{band} {lo:g}–{hi:g} Hz"


def kept_text(dev: dict) -> str:
    drops = [f"{dev[k]} {w}" for k, w in (("artifact", "artifact"), ("missing", "no data / gap")) if dev[k]]
    return f"{dev['kept']} of {dev['total']} kept" + (f" ({', '.join(drops)} dropped)" if drops else "")


def classifier_text(c: dict) -> str:
    if c["error"]:
        return c["error"]
    return (f"{100 * c['acc']:.0f}% vs chance {100 * c['chance']:.0f}% (p = {_p(c['p'])}; "
            f"{c['n']} trials, {c['features']} features, {c['folds']}-fold, {len(c['null'])} shuffles)")


def report(res: dict) -> str:
    """The analysis as text (scope.py --report)."""
    classes = res["classes"]
    lines = [f"Cued trials ({res['task'] or 'task ?'}): {res['n_trials']} trials, "
             + ", ".join(f"{c} {res['by_class'][c]}" for c in classes)
             + f"; reference: {REFERENCE_LABEL[res['reference']]}"]
    for dev in res["devices"].values():
        lines.append(f"  {dev['label']}: {kept_text(dev)}")
    (b0, b1), (t0, t1) = res["windows"]["baseline"], res["windows"]["task"]
    lines.append(f"ERD in dB (power {t0:g}–{t1:g} s after the cue vs {b0:g}–{b1:g} s; negative = "
                 "desynchronised): mean ± SE (trials), * p<0.05 ** p<0.01 vs 0")
    w = 22
    for band, _ in BANDS:
        lines.append("")
        head = f"  {band_label(band)}".ljust(24) + "".join(c.rjust(w) for c in classes)
        lines.append(head + ("left vs right p".rjust(18) if "left" in classes and "right" in classes else ""))
        for e in res["electrodes"]:
            row = f"  {e['position']:<5} {e['label']} {e['sdk']}".ljust(24)
            row += "".join(_cell(e["erd"][band][c]).rjust(w) for c in classes)
            if e["lr"][band] is not None:
                row += _p(e["lr"][band]["p"]).rjust(18)
            lines.append(row)
    lines.append("")
    lat = res["lateralisation"]
    if lat is None:
        lines.append("Lateralisation (C4 − C3): needs leads at C3 and C4")
    else:
        lines.append("Lateralisation C4 − C3 ERD (dB; left hand expected < 0, right hand > 0)")
        for band, _ in BANDS:
            row = f"  {band:<5}" + "".join(f"   {c} {_cell(lat[band]['by_class'][c])}" for c in classes)
            if lat[band]["lr"] is not None:
                row += f"   left vs right p = {_p(lat[band]['lr']['p'])}"
            lines.append(row)
    lines.append("")
    pair = next((c["classes"] for c in res["classifier"].values() if c["classes"]), [])
    lines.append(f"Classifier ({' vs '.join(pair) or 'classes'}; shrinkage LDA on each trial's log band power "
                 f"{t0:g}–{t1:g} s after the cue, stratified K-fold, labels shuffled for chance)")
    for name, c in res["classifier"].items():
        lines.append(f"  {res['devices'][name]['label']:<10} {classifier_text(c)}")
    return "\n".join(lines)
