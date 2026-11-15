"""The scope window's cued-trial parts, mixed into app.ScopeApp:

    T in LIVE      asks for the task (imagine / squeeze), trials per class and a name, then records a
                   cued session full-window (fixation cross, ← / → / ○ cue, pause), saves it with its
                   trial markers and opens its analysis. Esc ends early and keeps the finished trials.
    TRIALS view    one recording's per-trial analysis (rawscope/trials.py): ERD per electrode and class,
                   the ERD time course, lateralisation and the classifier against shuffled labels.
"""
from __future__ import annotations

import dataclasses
import math
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pygame

from eeg import cues, recording
from ui import sound, theme

from . import trials

# Checked as a categorical set on the scope's dark panels (lightness band, chroma, colour-blind separation).
CLASS_COLORS = {"left": (57, 135, 224), "right": (217, 118, 26), "rest": (154, 123, 216)}
OTHER_COLOR = (25, 158, 112)
CLASS_TEXT = {"left": "left hand", "right": "right hand", "rest": "rest"}
PER_CLASS_STEP = 5
PER_CLASS_RANGE = (5, 40)
CUE_BEEP = (880.0, 150)
BREAK_BEEP = (523.0, 350)
SHADE = (40, 46, 60)


def class_color(label: str) -> tuple[int, int, int]:
    return CLASS_COLORS.get(label, OTHER_COLOR)


def _mmss(seconds: float) -> str:
    s = int(math.ceil(max(0.0, seconds)))
    return f"{s // 60}:{s % 60:02d}"


def _p_text(p: float) -> str:
    if not math.isfinite(p):
        return "–"
    return f"{p:.3g}" if p >= 0.001 else "<0.001"


def _p_is(p: float) -> str:
    """"p = 0.0123" or "p < 0.001"."""
    txt = _p_text(p)
    return f"p {txt[0]} {txt[1:]}" if txt.startswith("<") else f"p = {txt}"


def _stars(p: float) -> str:
    return "**" if p < 0.01 else "*" if p < 0.05 else ""


def _scale(values, floor: float = 2.0, cap: float = 10.0) -> float:
    """A symmetric dB range that holds every finite value."""
    finite = [abs(v) for v in values if v is not None and math.isfinite(v)]
    return float(min(cap, max(floor, math.ceil(max(finite) if finite else 0.0))))


def _arrow(cx: float, cy: float, s: float, direction: int) -> list[tuple[float, float]]:
    pts = [(-1.0, -0.17), (0.12, -0.17), (0.12, -0.5), (1.0, 0.0), (0.12, 0.5), (0.12, 0.17), (-1.0, 0.17)]
    return [(cx + direction * x * s, cy + y * s) for x, y in pts]


class TrialsUI:
    def _init_trials(self, protocol: cues.Protocol) -> None:
        self.protocol = protocol                     # defaults for the next session
        self.cued: Optional[cues.Run] = None
        self.trial_view: Optional[dict] = None
        self._pool: Optional[ThreadPoolExecutor] = None
        self._tip: Optional[tuple[tuple[int, int], str]] = None

    # -- a cued session --------------------------------------------------------------------
    def open_cued_prompt(self) -> None:
        if self.feeds.recording:
            self.flash("Stop the recording first (Space).", theme.WARN)
            return
        self.picker = None
        self.prompt = {"kind": "cued", "title": "Cued trials", "text": "", "task": self.protocol.task,
                       "per_class": self.protocol.per_class, "swallow": ("t", time.monotonic() + 0.25)}

    def _cued_prompt_key(self, k: int) -> bool:
        p = self.prompt
        if k in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self.prompt = None
            self.start_cued(p["task"], p["per_class"], p["text"].strip())
        elif k in (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_TAB):
            p["task"] = cues.TASKS[(cues.TASKS.index(p["task"]) + 1) % len(cues.TASKS)]
        elif k == pygame.K_UP:
            p["per_class"] = min(PER_CLASS_RANGE[1], p["per_class"] + PER_CLASS_STEP)
        elif k == pygame.K_DOWN:
            p["per_class"] = max(PER_CLASS_RANGE[0], p["per_class"] - PER_CLASS_STEP)
        else:
            return False
        return True

    def start_cued(self, task: str = "imagine", per_class: Optional[int] = None, name: str = "") -> bool:
        if self.feeds.recording:
            self.flash("Stop the recording first (Space).", theme.WARN)
            return False
        if not any(d.state == "streaming" and not d.silent(self.now) for d in self.feeds.devices):
            self.flash("Nothing is streaming yet: wait for a device to connect.", theme.WARN, 4.0)
            return False
        proto = dataclasses.replace(self.protocol, task=task, per_class=per_class or self.protocol.per_class)
        self.feeds.start_recording()
        self.cued = cues.Run(proto, self.feeds.recorder.start, name=name or f"cued {task}")
        self.set_view("live")
        return True

    def end_cued(self) -> None:
        """Esc: stop now, save the trials finished so far."""
        if self.cued is not None:
            self.cued.stop()
            self._finish_cued()

    def _update_cued(self) -> None:
        run = self.cued
        for step in run.update(self.feeds.clock()):
            if step.kind == "cue":
                sound.beep(*CUE_BEEP)
            elif step.kind in ("break", "done"):
                sound.beep(*BREAK_BEEP)
        s = run.step
        self.feeds.mock_cue(s.label if s.kind == "cue" and not run.finished else None)
        if run.finished:
            self._finish_cued()

    def _finish_cued(self) -> None:
        run = self.cued
        path = self.stop_recording(ask_name=False)          # saves run.meta() with it
        if path is None:
            return
        n = run.completed
        self.flash(f"Saved {n} trial{'' if n == 1 else 's'}" + (" (ended early)" if run.ended_early else "")
                   + f": {path.name}", theme.TEXT, 4.0)
        if n:
            self.open_trials(path)

    def _cued_key(self, k: int) -> None:
        if k == pygame.K_ESCAPE:
            self.end_cued()
        elif k == pygame.K_SPACE:
            self.cued.skip(self.feeds.clock())

    # -- the TRIALS view: state ---------------------------------------------------------------
    def open_trials(self, path, switch: bool = True) -> bool:
        path = Path(path)
        try:
            rec = recording.load(path)
        except Exception as exc:
            self.flash(f"Couldn't open {path.name}: {exc}", theme.BAD)
            return False
        if not trials.has_trials(rec["meta"]):
            self.flash(f"{rec['meta'].get('name') or path.name} has no cued trials (T in LIVE records some).",
                       theme.WARN, 4.0)
            return False
        old = self.trial_view
        self.trial_view = {"path": path, "rec": rec, "ref": old["ref"] if old else trials.REFERENCES[0],
                           "jobs": {}, "focus": old["focus"] if old else None}
        if switch:
            self.set_view("trials")
        return True

    def trial_result(self, ref: Optional[str] = None, wait: bool = False) -> Optional[dict]:
        """The analysis of the open recording (computed in the background: the classifier's label
        shuffles take a second or two). None while it's still running."""
        tv = self.trial_view
        if tv is None:
            return None
        ref = ref or tv["ref"]
        job: Optional[Future] = tv["jobs"].get(ref)
        if job is None:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="trials")
            job = tv["jobs"][ref] = self._pool.submit(trials.analyse, tv["rec"], ref)
        if not wait and not job.done():
            return None
        try:
            return job.result()
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    def toggle_reference(self) -> None:
        tv = self.trial_view
        if tv is not None:
            tv["ref"] = trials.REFERENCES[(trials.REFERENCES.index(tv["ref"]) + 1) % len(trials.REFERENCES)]

    def _set_focus(self, position: str) -> None:
        if self.trial_view is not None:
            self.trial_view["focus"] = position

    def _move_focus(self, step: int) -> None:
        res = self.trial_result()
        if not res or "error" in res or not res["electrodes"]:
            return
        names = [e["position"] for e in res["electrodes"]]
        i = names.index(self.trial_view["focus"]) if self.trial_view["focus"] in names else 0
        self.trial_view["focus"] = names[(i + step) % len(names)]

    def _on_trials_key(self, k: int) -> None:
        if k == pygame.K_a:
            self.toggle_reference()
        elif k == pygame.K_LEFT:
            self._move_focus(-1)
        elif k == pygame.K_RIGHT:
            self._move_focus(1)
        elif k == pygame.K_p and self.trial_view is not None:
            self.open_playback(self.trial_view["path"])

    def _close_trials_pool(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    # -- drawing: the setup prompt -----------------------------------------------------------
    def _draw_cued_prompt(self, surf) -> None:
        f = self.fonts
        p = self.prompt
        w, h = surf.get_size()
        self._shade(surf)
        box = pygame.Rect(0, 0, f.px(720), f.px(330))
        box.center = (w // 2, h // 2)
        theme.panel(surf, box, theme.PANEL, theme.EDGE, f.px(12))
        pad = f.px(24)
        x, y = box.x + pad, box.y + f.px(22)
        f.text(surf, "Cued trials: left hand, right hand, rest", 22, theme.TEXT, (x, y), bold=True)
        y += f.px(34)
        pr = self.protocol
        f.text(surf, f"Fixation cross {pr.baseline_s:g} s, then an arrow (or a circle = rest) for {pr.cue_s:g} s, "
                     f"then a {pr.pause_s[0]:g}-{pr.pause_s[1]:g} s pause. A beep marks each cue.",
               15, theme.MUTED, (x, y), max_width=box.w - 2 * pad)
        y += f.px(36)
        f.text(surf, "Task", 16, theme.MUTED, (x, y + f.px(4)), bold=True)
        cx = x + f.px(130)
        for task in cues.TASKS:
            on = task == p["task"]
            txt = f.render(task, 17, theme.INK if on else theme.MUTED, bold=True)
            chip = pygame.Rect(cx, y, txt.get_width() + f.px(28), f.px(30))
            theme.panel(surf, chip, theme.TEXT if on else theme.PANEL_HI, None if on else theme.EDGE, f.px(15))
            surf.blit(txt, txt.get_rect(center=chip.center))
            self._hover(chip)
            self._modal_hits.append((chip, lambda t=task: p.__setitem__("task", t)))
            cx = chip.right + f.px(8)
        f.text(surf, "←→", 14, theme.DIM, (cx + f.px(6), y + f.px(15)), "midleft")
        y += f.px(36)
        f.text(surf, cues.TASK_TEXT[p["task"]], 15, theme.TEXT, (x + f.px(130), y), max_width=box.w - 2 * pad - f.px(130))
        y += f.px(34)
        proto = dataclasses.replace(self.protocol, per_class=p["per_class"])
        n = proto.n_trials
        per_trial = proto.baseline_s + proto.cue_s + sum(proto.pause_s) / 2
        total = proto.lead_in_s + (proto.practice + n) * per_trial + proto.tail_s + \
            (max(0, (n - 1) // proto.break_every) * proto.break_s if proto.break_every else 0)
        f.text(surf, "Trials", 16, theme.MUTED, (x, y), bold=True)
        f.text(surf, f"{p['per_class']} per class = {n}  (↑↓)   ·   about {_mmss(total)} with "
                     f"{proto.practice} practice trial{'' if proto.practice == 1 else 's'} and breaks", 16,
               theme.TEXT, (x + f.px(130), y),
               max_width=box.w - 2 * pad - f.px(130))
        y += f.px(40)
        f.text(surf, "Name", 16, theme.MUTED, (x, y + f.px(12)), bold=True)
        field = pygame.Rect(x + f.px(130), y, box.w - 2 * pad - f.px(130), f.px(44))
        self._text_field(surf, field, p["text"], f"optional, e.g. sam-{p['task']}-1")
        f.text(surf, "Enter start (records until done) · Esc cancel · during the run: Esc ends early, "
                     "Space skips a break", 14, theme.MUTED, (x, box.bottom - f.px(24)), "midleft",
               max_width=box.w - 2 * pad)

    # -- drawing: the session ---------------------------------------------------------------------
    def _draw_cued(self, surf) -> None:
        f = self.fonts
        run = self.cued
        w, h = surf.get_size()
        surf.fill(theme.BG)
        now = self.feeds.clock()
        s = run.step
        n = run.protocol.n_trials
        unit = min(w, h)
        cx, cy = w / 2, h / 2
        # top: where we are, how long is left (dim, so it doesn't pull the eyes)
        if s.practice:
            where = "PRACTICE"
        elif s.trial is not None:
            where = f"TRIAL {s.trial + 1} / {n}"
        else:
            where = f"{run.completed} / {n} TRIALS DONE"
        f.text(surf, where, 17, theme.MUTED, (f.px(24), f.px(26)), "midleft", bold=True)
        f.text(surf, f"{_mmss(run.remaining(now))} left   ·   Esc ends early (keeps finished trials)", 15,
               theme.DIM, (w - f.px(24), f.px(26)), "midright")
        f.text(surf, f"{run.protocol.task.upper()}  ·  {cues.TASK_TEXT[run.protocol.task]}", 15, theme.DIM,
               (cx, f.px(26)), "center")
        total = sum(st.seconds for st in run.steps)
        bar = pygame.Rect(0, f.px(48), w, f.px(3))
        pygame.draw.rect(surf, theme.PANEL_HI, bar)
        bar.w = round(w * min(1.0, 1.0 - run.remaining(now) / max(total, 1e-9)))
        pygame.draw.rect(surf, theme.EDGE, bar)

        left = max(0.0, s.seconds - run.elapsed_in_step(now))
        if s.kind == "fixation":
            self._draw_cross(surf, cx, cy, unit)
        elif s.kind == "cue":
            size = unit * 0.17
            if s.label in ("left", "right"):
                pygame.draw.polygon(surf, theme.TEXT, _arrow(cx, cy, size, -1 if s.label == "left" else 1))
            else:
                pygame.draw.circle(surf, theme.TEXT, (round(cx), round(cy)), round(size * 0.55), f.px(10))
                f.text(surf, s.label, 22, theme.MUTED, (cx, cy + size * 0.8), "midtop", bold=True)
        elif s.kind == "pause":
            f.text(surf, "blink · relax", 18, theme.DIM, (cx, cy), "center")
        else:
            title, lines = self._cued_message(s.kind, run, left)
            f.text(surf, title, 44, theme.TEXT, (cx, cy - f.px(90)), "midbottom", bold=True)
            y = cy - f.px(60)
            for line, col in lines:
                for part in theme.wrap(f, line, 20, round(w * 0.6)):
                    f.text(surf, part, 20, col, (cx, y), "midtop")
                    y += f.px(30)
                y += f.px(8)
        # a device that stopped streaming: say so, small, at the bottom
        warn = []
        for d in self.feeds.devices:
            if d.state != "streaming":
                warn.append(f"{d.label}: {d.state}")
            elif d.silent(self.now):
                warn.append(f"{d.label}: no samples")
        if warn:
            msg = "   ·   ".join(warn) + "   (still recording)"
            r = f.render(msg, 16, theme.WARN, bold=True).get_rect(midbottom=(round(cx), h - f.px(22)))
            theme.dot(surf, theme.WARN, (r.x - f.px(14), r.centery), f.px(6))
            f.text(surf, msg, 16, theme.WARN, r.midbottom, "midbottom", bold=True)

    def _draw_cross(self, surf, cx: float, cy: float, unit: float) -> None:
        a, t = unit * 0.045, self.fonts.px(6)
        pygame.draw.rect(surf, theme.TEXT, (round(cx - a), round(cy - t / 2), round(2 * a), t))
        pygame.draw.rect(surf, theme.TEXT, (round(cx - t / 2), round(cy - a), t, round(2 * a)))

    @staticmethod
    def _cued_message(kind: str, run: cues.Run, left: float) -> tuple[str, list]:
        p = run.protocol
        verb = "imagine squeezing" if p.task == "imagine" else "squeeze"
        if kind == "ready":
            first = (f"First {p.practice} practice trial{'' if p.practice == 1 else 's'}." if p.practice else "")
            return "Get ready", [
                (f"Keep your eyes on the cross. Arrow ← or →: {verb} that hand until it goes. "
                 "Circle: just rest.", theme.TEXT),
                ("Stay still, jaw relaxed; blink in the pauses, not during the arrow. " + first, theme.MUTED),
                (f"Starting in {math.ceil(left)}  ·  Space starts now", theme.DIM)]
        if kind == "go":
            return "Practice done", [(f"Now the real thing: {p.n_trials} trials.", theme.TEXT),
                                     (f"Starting in {math.ceil(left)}", theme.DIM)]
        if kind == "break":
            return "Break", [(f"{run.completed} of {p.n_trials} done. Rest your hands, blink, move a little.",
                              theme.TEXT), (f"Next block in {math.ceil(left)}  ·  Space continues", theme.DIM)]
        return "Done", [("Saving…", theme.MUTED)]

    # -- drawing: the TRIALS view ----------------------------------------------------------------
    def _draw_trials(self, surf, body: pygame.Rect) -> None:
        f = self.fonts
        tv = self.trial_view
        self._tip = None
        if tv is None:
            return
        meta = tv["rec"]["meta"]
        proto = meta.get("protocol") or {}
        top = pygame.Rect(body.x, body.y, body.w, f.px(40))
        created = datetime.fromtimestamp(meta.get("created", 0))
        x = f.text(surf, meta.get("name") or "untitled", 22, theme.TEXT, (top.x + f.px(4), top.centery), "midleft",
                   bold=True).right
        sub = f"{proto.get('task', '?')}  ·  {created:%b %d %H:%M}  ·  {_mmss(meta.get('duration_s', 0))}" + \
              ("  ·  ended early" if proto.get("ended_early") else "") + ("  ·  mock" if meta.get("mock") else "")
        x = f.text(surf, sub, 15, theme.MUTED, (x + f.px(14), top.centery), "midleft").right
        # reference chips, right
        rx = top.right
        for ref in reversed(trials.REFERENCES):
            on = ref == tv["ref"]
            txt = f.render(trials.REFERENCE_LABEL[ref], 15, theme.INK if on else theme.MUTED, bold=on)
            chip = pygame.Rect(0, 0, txt.get_width() + f.px(22), f.px(30))
            chip.midright = (rx, top.centery)
            theme.panel(surf, chip, theme.TEXT if on else theme.PANEL, None if on else theme.EDGE, f.px(15))
            surf.blit(txt, txt.get_rect(center=chip.center))
            self._hover(chip)
            self._hits.append((chip, lambda r=ref: tv.__setitem__("ref", r)))
            rx = chip.x - f.px(6)
        f.text(surf, "Reference (A)", 15, theme.DIM, (rx - f.px(6), top.centery), "midright")

        res = self.trial_result()
        if res is None:
            f.text(surf, f"Analysing ({trials.REFERENCE_LABEL[tv['ref']]}): filtering every trial, then the "
                         f"classifier against {trials.N_PERM} label shuffles…", 18, theme.MUTED, body.center, "center")
            return
        if "error" in res:
            f.text(surf, f"Couldn't analyse: {res['error']}", 18, theme.BAD, body.center, "center",
                   max_width=body.w - f.px(40))
            return
        if tv["focus"] not in [e["position"] for e in res["electrodes"]] and res["electrodes"]:
            tv["focus"] = res["electrodes"][0]["position"]
        y = top.bottom + f.px(4)
        parts = [f"{res['n_trials']} trials (" + ", ".join(f"{res['by_class'][c]} {c}" for c in res["classes"]) + ")"]
        parts += [f"{d['label']}: {trials.kept_text(d)}" for d in res["devices"].values()]
        f.text(surf, "   ·   ".join(parts), 15, theme.MUTED, (body.x + f.px(4), y), max_width=body.w - f.px(8))
        y += f.px(26)
        lx = body.x + f.px(4)
        for c in res["classes"]:
            pygame.draw.rect(surf, class_color(c), (lx, y + f.px(4), f.px(14), f.px(12)), border_radius=f.px(3))
            lx = f.text(surf, CLASS_TEXT.get(c, c), 15, theme.TEXT, (lx + f.px(20), y), bold=True).right + f.px(18)
        (b0, _), (t0, t1) = res["windows"]["baseline"], res["windows"]["task"]
        f.text(surf, f"ERD = band power {t0:g}–{t1:g} s after the cue vs the {-b0:g} s before it (dB); "
                     "below 0 = desynchronised. Bars: mean ± SE over "
                     "clean trials, * p<0.05 ** p<0.01 vs 0. Hover for numbers.", 14, theme.DIM, (lx + f.px(8), y + f.px(1)),
               max_width=body.right - lx - f.px(12))
        main = pygame.Rect(body.x, y + f.px(28), body.w, body.bottom - y - f.px(28))
        left = pygame.Rect(main.x, main.y, round(main.w * 0.55), main.h)
        right = pygame.Rect(left.right + f.px(12), main.y, main.right - left.right - f.px(12), main.h)
        self._draw_erd_grid(surf, left, res)
        gap = f.px(10)
        h1, h2 = round(right.h * 0.40), round(right.h * 0.26)
        self._draw_course(surf, pygame.Rect(right.x, right.y, right.w, h1 - gap), res)
        self._draw_lateralisation(surf, pygame.Rect(right.x, right.y + h1, right.w, h2 - gap), res)
        self._draw_classifier(surf, pygame.Rect(right.x, right.y + h1 + h2, right.w, right.h - h1 - h2), res)
        if self._tip is not None:
            (tx, ty), text = self._tip
            r = f.render(text, 15, theme.TEXT, bold=True).get_rect()
            r.midbottom = (tx, ty - f.px(10))
            r.clamp_ip(surf.get_rect().inflate(-f.px(20), -f.px(20)))
            theme.panel(surf, r.inflate(f.px(18), f.px(10)), theme.PANEL_HI, theme.MUTED, f.px(6))
            f.text(surf, text, 15, theme.TEXT, r.topleft, bold=True)

    def _bar_group(self, surf, r: pygame.Rect, values: list, ymax: float, what: str) -> None:
        """One bar per class (mean ± SE, dB) around a zero line at r.centery; hover sets the tooltip."""
        f = self.fonts
        n = max(1, len(values))
        gap = f.px(4)
        bw = max(f.px(6), min(f.px(28), (r.w - gap * (n - 1)) / n))
        x0 = r.centerx - (n * bw + (n - 1) * gap) / 2
        y0 = r.centery
        k = (r.h / 2 - f.px(16)) / ymax

        def yv(v: float) -> float:
            return y0 - max(-ymax, min(ymax, v)) * k

        for i, (label, s) in enumerate(values):
            bx = x0 + i * (bw + gap)
            hit = pygame.Rect(round(bx - gap / 2), r.y, round(bw + gap), r.h)
            if hit.collidepoint(self._mouse) and self._no_modal():
                pygame.draw.rect(surf, theme.PANEL_HI, hit, border_radius=f.px(4))
                tip = f"{what} · {CLASS_TEXT.get(label, label)}: "
                tip += "no clean trials" if not math.isfinite(s["mean"]) else \
                    f"{s['mean']:+.2f}" + (f" ± {s['se']:.2f}" if math.isfinite(s["se"]) else "") + \
                    f" dB, {s['n']} trials, {_p_is(s['p'])}"
                self._tip = ((hit.centerx, r.y), tip)
        pygame.draw.line(surf, theme.EDGE, (r.x, y0), (r.right, y0), 1)
        for i, (label, s) in enumerate(values):
            bx = x0 + i * (bw + gap)
            m = s["mean"]
            if not math.isfinite(m):
                f.text(surf, "–", 13, theme.DIM, (bx + bw / 2, y0), "center")
                continue
            top, bottom = sorted((yv(m), y0))
            rect = pygame.Rect(round(bx), round(top), round(bw), max(1, round(bottom - top)))
            rad = min(f.px(4), rect.h, round(bw / 2))
            if m >= 0:
                pygame.draw.rect(surf, class_color(label), rect, border_top_left_radius=rad, border_top_right_radius=rad)
            else:
                pygame.draw.rect(surf, class_color(label), rect, border_bottom_left_radius=rad,
                                 border_bottom_right_radius=rad)
            cxb = bx + bw / 2
            end = yv(m)
            if math.isfinite(s["se"]):
                y1, y2 = yv(m + s["se"]), yv(m - s["se"])
                pygame.draw.line(surf, theme.TEXT, (cxb, y1), (cxb, y2), 1)
                for yy in (y1, y2):
                    pygame.draw.line(surf, theme.TEXT, (cxb - bw * 0.2, yy), (cxb + bw * 0.2, yy), 1)
                end = y2 if m < 0 else y1
            stars = _stars(s["p"])
            if stars:
                f.text(surf, stars, 15, theme.TEXT, (cxb, end + (f.px(1) if m < 0 else -f.px(1))),
                       "midtop" if m < 0 else "midbottom", bold=True)

    def _db_grid(self, surf, r: pygame.Rect, ymax: float, labels: bool) -> None:
        f = self.fonts
        step = 1 if ymax <= 4 else 2 if ymax <= 8 else 5
        v = step
        while v <= ymax + 1e-9:
            for sign in (1, -1):
                y = r.centery - sign * v / ymax * (r.h / 2 - f.px(16))
                pygame.draw.line(surf, theme.PANEL_HI, (r.x, y), (r.right, y), 1)
                if labels:
                    f.text(surf, f"{sign * v:+g}", 11, theme.DIM, (r.x - f.px(3), y), "midright")
            v += step

    def _draw_erd_grid(self, surf, r: pygame.Rect, res: dict) -> None:
        f = self.fonts
        els = res["electrodes"]
        if not els:
            f.text(surf, "No electrodes in this recording.", 18, theme.DIM, r.center, "center")
            return
        classes = res["classes"]
        ymax = _scale(abs(s["mean"]) + (s["se"] if math.isfinite(s["se"]) else 0.0)
                      for e in els for band in e["erd"].values() for s in band.values() if math.isfinite(s["mean"]))
        cols = min(4, len(els))
        rows = math.ceil(len(els) / cols)
        gap = f.px(10)
        pw, ph = (r.w - gap * (cols - 1)) / cols, (r.h - gap * (rows - 1)) / rows
        for k, e in enumerate(els):
            pr = pygame.Rect(round(r.x + (k % cols) * (pw + gap)), round(r.y + (k // cols) * (ph + gap)),
                             round(pw), round(ph))
            focus = e["position"] == self.trial_view["focus"]
            hover = self._hover(pr) and not focus
            theme.panel(surf, pr, theme.PANEL, theme.TEXT if focus else theme.MUTED if hover else theme.EDGE,
                        f.px(8), 2 if focus else 1)
            self._hits.append((pr, lambda p=e["position"]: self._set_focus(p)))
            pad = f.px(10)
            f.text(surf, e["position"], 20, theme.TEXT, (pr.x + pad, pr.y + f.px(6)), bold=True)
            f.text(surf, f"{e['label']} {e['sdk']} · {e['kept']} clean", 13, theme.DIM,
                   (pr.right - pad, pr.y + f.px(10)), "topright", max_width=pr.w - f.px(60))
            chart = pygame.Rect(pr.x + pad + f.px(20), pr.y + f.px(36), pr.w - 2 * pad - f.px(20),
                                pr.h - f.px(36) - f.px(44))
            self._db_grid(surf, chart, ymax, labels=True)
            gw = chart.w / len(trials.BANDS)
            for j, (band, _) in enumerate(trials.BANDS):
                gr = pygame.Rect(round(chart.x + j * gw), chart.y, round(gw), chart.h)
                self._bar_group(surf, gr.inflate(-f.px(6), 0), [(c, e["erd"][band][c]) for c in classes], ymax,
                                f"{e['position']} {band}")
                f.text(surf, band, 14, theme.MUTED, (gr.centerx, chart.bottom + f.px(4)), "midtop", bold=True)
                lr = e["lr"][band]
                if lr is not None and math.isfinite(lr["p"]):
                    col = theme.TEXT if lr["p"] < 0.05 else theme.DIM
                    f.text(surf, f"L vs R p {_p_text(lr['p'])}", 12, col, (gr.centerx, chart.bottom + f.px(22)),
                           "midtop", max_width=round(gw))

    def _draw_course(self, surf, r: pygame.Rect, res: dict) -> None:
        f = self.fonts
        theme.panel(surf, r, theme.PANEL, theme.EDGE, f.px(8))
        e = next((x for x in res["electrodes"] if x["position"] == self.trial_view["focus"]), None)
        if e is None:
            return
        pad = f.px(12)
        f.text(surf, f"{e['position']}  over time", 18, theme.TEXT, (r.x + pad, r.y + f.px(8)), bold=True)
        f.text(surf, "dB vs its baseline, mean per class · shaded = task window · ←→ / click: electrode", 13,
               theme.DIM, (r.right - pad, r.y + f.px(12)), "topright", max_width=r.w - f.px(170))
        times = res["times"]
        classes = res["classes"]
        ymax = _scale(float(np.nanmax(np.abs(m))) for band in e["course"].values()
                      for m, _ in band.values() if np.isfinite(m).any())
        area = pygame.Rect(r.x + pad + f.px(22), r.y + f.px(40), r.w - 2 * pad - f.px(22), r.h - f.px(40) - f.px(30))
        pw = (area.w - f.px(30)) / len(trials.BANDS)
        t0, t1 = float(times[0]), float(times[-1])
        for j, (band, _) in enumerate(trials.BANDS):
            pr = pygame.Rect(round(area.x + j * (pw + f.px(30))), area.y, round(pw), area.h)

            def px(t: float) -> float:
                return pr.x + (t - t0) / (t1 - t0) * pr.w

            def py(v: float) -> float:
                return pr.centery - max(-ymax, min(ymax, v)) / ymax * (pr.h / 2 - f.px(16))

            pygame.draw.rect(surf, theme.PANEL_HI, pr, border_radius=f.px(4))
            a, b = res["windows"]["task"]
            pygame.draw.rect(surf, SHADE, (px(a), pr.y, px(b) - px(a), pr.h))
            self._db_grid(surf, pr, ymax, labels=True)
            pygame.draw.line(surf, theme.EDGE, (pr.x, pr.centery), (pr.right, pr.centery), 1)
            pygame.draw.line(surf, theme.MUTED, (px(0), pr.y), (px(0), pr.bottom), 1)
            for tick in range(math.ceil(t0), math.floor(t1) + 1):
                if tick % 2 == 0:
                    f.text(surf, f"{tick:+d} s" if tick else "cue", 12, theme.DIM, (px(tick), pr.bottom + f.px(3)),
                           "midtop")
            f.text(surf, trials.band_label(band), 14, theme.MUTED, (pr.x + f.px(6), pr.y + f.px(4)), bold=True)
            for c in classes:
                mean, _ = e["course"][band][c]
                ok = np.isfinite(mean)
                xs = [px(t) for t in times]
                start = None
                for i in range(len(times) + 1):
                    if i < len(times) and ok[i]:
                        start = i if start is None else start
                    elif start is not None:
                        if i - start >= 2:
                            pygame.draw.lines(surf, class_color(c), False,
                                              [(xs[q], py(mean[q])) for q in range(start, i)], 2)
                        start = None
            if pr.collidepoint(self._mouse) and self._no_modal():
                i = int(np.argmin(np.abs(np.array([px(t) for t in times]) - self._mouse[0])))
                pygame.draw.line(surf, theme.TEXT, (px(times[i]), pr.y), (px(times[i]), pr.bottom), 1)
                vals = "  ".join(f"{c} {e['course'][band][c][0][i]:+.1f}" for c in classes
                                 if math.isfinite(e["course"][band][c][0][i]))
                self._tip = ((round(px(times[i])), pr.y), f"{band} at {times[i]:+.1f} s: {vals or 'no data'} dB")

    def _draw_lateralisation(self, surf, r: pygame.Rect, res: dict) -> None:
        f = self.fonts
        theme.panel(surf, r, theme.PANEL, theme.EDGE, f.px(8))
        pad = f.px(12)
        f.text(surf, "Lateralisation  C4 − C3", 18, theme.TEXT, (r.x + pad, r.y + f.px(8)), bold=True)
        f.text(surf, "ERD difference per trial (dB): left hand should go below 0, right hand above",
               13, theme.DIM, (r.right - pad, r.y + f.px(12)), "topright", max_width=r.w - f.px(230))
        lat = res["lateralisation"]
        if lat is None:
            f.text(surf, "Needs leads at C3 and C4: click a lead's name in LIVE to say where it is.", 15,
                   theme.DIM, (r.centerx, r.centery + f.px(8)), "center", max_width=r.w - 2 * pad)
            return
        classes = res["classes"]
        ymax = _scale(abs(s["mean"]) + (s["se"] if math.isfinite(s["se"]) else 0.0)
                      for band in lat.values() for s in band["by_class"].values() if math.isfinite(s["mean"]))
        chart = pygame.Rect(r.x + pad + f.px(22), r.y + f.px(36), r.w - 2 * pad - f.px(22), r.h - f.px(36) - f.px(26))
        self._db_grid(surf, chart, ymax, labels=True)
        gw = chart.w / len(trials.BANDS)
        for j, (band, _) in enumerate(trials.BANDS):
            gr = pygame.Rect(round(chart.x + j * gw), chart.y, round(gw), chart.h)
            self._bar_group(surf, gr.inflate(-round(gw * 0.3), 0), [(c, lat[band]["by_class"][c]) for c in classes],
                            ymax, f"C4 − C3 {band}")
            lr = lat[band]["lr"]
            text = band + (f"  ·  left vs right {_p_is(lr['p'])}" if lr is not None else "")
            col = theme.TEXT if lr is not None and lr["p"] < 0.05 else theme.MUTED
            f.text(surf, text, 14, col, (gr.centerx, chart.bottom + f.px(4)), "midtop", bold=True)

    def _draw_classifier(self, surf, r: pygame.Rect, res: dict) -> None:
        f = self.fonts
        theme.panel(surf, r, theme.PANEL, theme.EDGE, f.px(8))
        pad = f.px(12)
        pair = next((c["classes"] for c in res["classifier"].values() if c["classes"]), [])
        f.text(surf, "Classifier  " + " vs ".join(pair), 18, theme.TEXT, (r.x + pad, r.y + f.px(8)), bold=True)
        f.text(surf, f"shrinkage LDA, log band power per lead × band; chance from {res['n_perm']} label shuffles",
               13, theme.DIM, (r.right - pad, r.y + f.px(12)), "topright", max_width=r.w - f.px(230))
        rows = list(res["classifier"].items())
        if not rows:
            return
        top = r.y + f.px(40)
        row_h = min(f.px(76), (r.bottom - top - f.px(18)) / len(rows))
        for i, (name, c) in enumerate(rows):
            y = top + i * row_h
            label = res["devices"][name]["label"]
            f.text(surf, label, 16, theme.TEXT, (r.x + pad, y + f.px(4)), bold=True)
            if c["error"]:
                f.text(surf, c["error"], 15, theme.DIM, (r.x + pad + f.px(110), y + f.px(5)))
                continue
            good = c["p"] < 0.05
            f.text(surf, f"{100 * c['acc']:.0f}%", 26, theme.TEXT, (r.x + pad, y + f.px(24)), bold=True)
            track = pygame.Rect(r.x + pad + f.px(110), round(y + f.px(6)), r.w - 2 * pad - f.px(110),
                                round(row_h - f.px(32)))
            pygame.draw.rect(surf, theme.PANEL_HI, track, border_radius=f.px(4))

            def ax(a: float) -> float:
                return track.x + a * track.w

            vals, counts = np.unique(np.round(c["null"], 6), return_counts=True)
            bw = max(f.px(2), track.w / max(1, c["n"]) - f.px(1))
            for v, k in zip(vals, counts):
                hgt = (track.h - f.px(4)) * k / counts.max()
                pygame.draw.rect(surf, theme.EDGE, (ax(v) - bw / 2, track.bottom - hgt, bw, hgt))
            pygame.draw.line(surf, theme.MUTED, (ax(c["chance"]), track.y), (ax(c["chance"]), track.bottom), 1)
            pygame.draw.rect(surf, theme.TEXT, (round(ax(c["acc"])) - f.px(1), track.y, f.px(3), track.h))
            verdict = "above chance" if good else "not above chance"
            f.text(surf, f"{verdict}  ·  chance {100 * c['chance']:.0f}%  ·  {_p_is(c['p'])}  ·  "
                         f"{c['n']} trials, {c['features']} features, {c['folds']}-fold", 13,
                   theme.GOOD if good else theme.MUTED, (track.x, track.bottom + f.px(3)), bold=good,
                   max_width=track.w)
            hit = track.inflate(0, f.px(6))
            if hit.collidepoint(self._mouse) and self._no_modal():
                self._tip = ((round(ax(c["acc"])), track.y), f"{label}: {100 * c['acc']:.0f}% correct; grey = "
                             f"accuracy with shuffled labels ({len(c['null'])} times), line = their mean")
