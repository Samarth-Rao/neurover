"""Cued trials: the protocol (what the subject sees, and when) and a run of it in progress. Shared:
the scope's motor sessions (← left hand, → right hand, ○ rest) and the rover's SSVEP calibration
(look at the ↑ ↓ ← → box, or keep looking at the rover).

A trial is a fixation cross (its own baseline, ~2 s), then the cue for a few seconds with a beep at
onset, then a jittered pause (blink, relax). The rover's calibration instead starts each trial by
looking away at the real rover (``away_s``, jittered by ``away_jitter_s``), then a 3-2-1 hand-over on
screen pointing at the next box (the fixation step), the box, and no pause: as when driving. The classes are interleaved at random, never more than
2 of a kind in a row, so slow drift (electrodes settling over a session) can't pass for a class
difference, which is what sank one block per condition.

Run.trials times are on the clock the samples are timestamped with (Feeds.clock, time.monotonic),
so a recording's meta["trials"] lines up with its t<k> arrays directly.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import numpy as np

CLASSES = ("left", "right", "rest")
TASKS = ("imagine", "squeeze")
TASK_TEXT = {"imagine": "imagine squeezing that hand: don't move it",
             "squeeze": "squeeze that hand, gently, until the arrow goes"}
PROTOCOL_FORMAT = "cued/1"
SKIPPABLE = ("ready", "go", "break")      # Space moves on


@dataclass(frozen=True)
class Protocol:
    task: str = "imagine"
    classes: tuple[str, ...] = CLASSES
    per_class: int = 15
    max_run: int = 2                  # never more than this many of one class in a row
    lead_in_s: float = 8.0            # instructions + countdown
    practice: int = 3                 # practice trials first (recorded, marked, not analysed); 0 = none
    away_s: float = 0.0               # look away (at the rover) first; 0 = no such step
    away_jitter_s: float = 0.0        # ...plus up to this much at random, so the cue can't be anticipated
    baseline_s: float = 2.0           # fixation cross before the cue; 0 = none
    cue_s: float = 4.0
    pause_s: tuple[float, float] = (1.5, 3.0)   # jittered, so the next cue can't be anticipated; (0, 0) = none
    break_every: int = 15             # trials between breaks (0 = no breaks)
    break_s: float = 20.0
    tail_s: float = 2.0               # after the last trial, so its post-cue data is recorded

    @property
    def n_trials(self) -> int:
        return self.per_class * len(self.classes)

    def to_meta(self) -> dict:
        return {"format": PROTOCOL_FORMAT, **asdict(self)}


@dataclass(frozen=True)
class Step:
    kind: str                         # ready, away, fixation, cue, pause, go, break, done
    seconds: float
    label: Optional[str] = None       # the trial's class (fixation / cue / pause)
    trial: Optional[int] = None       # 0-based among the real trials (None for practice)
    practice: bool = False


def sequence(classes: Sequence[str], per_class: int, max_run: int = 2,
             rng: Optional[np.random.Generator] = None) -> list[str]:
    """``per_class`` of each class in random order, no more than ``max_run`` of one in a row.
    Picks are weighted by how many of each are left, so the classes stay mixed all the way through."""
    rng = rng if rng is not None else np.random.default_rng()
    classes = list(classes)
    if len(classes) == 1 and per_class > max_run:
        raise ValueError("one class can't be interleaved")
    for _ in range(1000):
        left = {c: per_class for c in classes}
        out: list[str] = []
        while len(out) < per_class * len(classes):
            ok = [c for c in classes if left[c] > 0 and out[-max_run:] != [c] * max_run]
            if not ok:
                break
            w = np.array([left[c] for c in ok], dtype=float)
            c = ok[int(rng.choice(len(ok), p=w / w.sum()))]
            out.append(c)
            left[c] -= 1
        else:
            return out
    raise ValueError("couldn't order the trials")


def plan(p: Protocol, rng: np.random.Generator) -> list[Step]:
    """Every step of a session, in order."""
    steps = [Step("ready", p.lead_in_s)]

    def trial(label: str, i: Optional[int], practice: bool) -> None:
        if p.away_s > 0:
            steps.append(Step("away", p.away_s + float(rng.uniform(0.0, p.away_jitter_s)), label, i, practice))
        if p.baseline_s > 0:
            steps.append(Step("fixation", p.baseline_s, label, i, practice))
        steps.append(Step("cue", p.cue_s, label, i, practice))
        if max(p.pause_s) > 0:
            steps.append(Step("pause", float(rng.uniform(*p.pause_s)), label, i, practice))

    if p.practice > 0:
        for label in rng.permutation([p.classes[k % len(p.classes)] for k in range(p.practice)]):
            trial(str(label), None, True)
        steps.append(Step("go", min(4.0, p.lead_in_s)))
    order = sequence(p.classes, p.per_class, p.max_run, rng)
    for i, label in enumerate(order):
        trial(label, i, False)
        if p.break_every and (i + 1) % p.break_every == 0 and i + 1 < len(order):
            steps.append(Step("break", p.break_s))
    steps.append(Step("done", p.tail_s))
    return steps


class Run:
    """A protocol being shown: which step it's on, and the trials finished so far.

    Steps keep to the plan's timing (no drift from frame timing); the markers are when each step
    actually appeared. A trial counts as finished when its cue ends."""

    def __init__(self, protocol: Protocol, start: float, name: str = "", seed: Optional[int] = None) -> None:
        self.protocol = protocol
        self.name = name
        self.seed = int(seed) if seed is not None else int(np.random.SeedSequence().entropy % 2 ** 31)
        self.steps = plan(protocol, np.random.default_rng(self.seed))
        self.start = start
        self.i = 0
        self.step_start = start
        self.trials: list[dict] = []
        self._open: Optional[dict] = None
        self.finished = False
        self.ended_early = False

    @property
    def step(self) -> Step:
        return self.steps[min(self.i, len(self.steps) - 1)]

    @property
    def completed(self) -> int:
        return sum(1 for t in self.trials if not t.get("practice"))

    def update(self, now: float) -> list[Step]:
        """Move on to ``now``; returns the steps that began (the caller beeps at a cue)."""
        began = []
        while not self.finished and now - self.step_start >= self.step.seconds:
            self.step_start += self.step.seconds
            self._leave(self.step, now)
            self.i += 1
            if self.i >= len(self.steps):
                self.finished = True
            else:
                self._enter(self.step, now)
                began.append(self.step)
        return began

    def _enter(self, step: Step, now: float) -> None:
        if step.kind in ("away", "fixation"):
            if step.kind == "away" or self._open is None:      # a trial opens at its first step
                self._open = {"label": step.label}
                if step.practice:
                    self._open["practice"] = True
            self._open["away" if step.kind == "away" else "baseline"] = round(now, 4)
        elif step.kind == "cue" and self._open is not None:
            self._open["cue"] = round(now, 4)

    def _leave(self, step: Step, now: float) -> None:
        if step.kind == "cue" and self._open is not None and "cue" in self._open:
            self._open["end"] = round(now, 4)
            self.trials.append(self._open)
            self._open = None

    def skip(self, now: float) -> bool:
        """Cut a break (or the lead-in) short."""
        if self.finished or self.step.kind not in SKIPPABLE:
            return False
        self.step_start = now - self.step.seconds
        return True

    def stop(self) -> None:
        """End early: the trial being shown (if its cue hasn't ended) is left out."""
        if not self.finished:
            self.finished = self.ended_early = True
            self._open = None

    def elapsed_in_step(self, now: float) -> float:
        return max(0.0, now - self.step_start)

    def remaining(self, now: float) -> float:
        if self.finished:
            return 0.0
        rest = sum(s.seconds for s in self.steps[self.i + 1:])
        return max(0.0, self.step.seconds - (now - self.step_start)) + rest

    def meta(self) -> dict:
        """What a recording keeps: meta["trials"] and meta["protocol"]."""
        return {"trials": [dict(t) for t in self.trials],
                "protocol": {**self.protocol.to_meta(), "name": self.name, "seed": self.seed,
                             "planned": self.protocol.n_trials, "completed": self.completed,
                             "ended_early": self.ended_early}}
