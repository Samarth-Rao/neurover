# neurover

Drive a small rover with EEG: the driver looks at one of four flickering boxes (SSVEP) and a
BrainBit Flex + BrainBit headband pick up which. A simulated rover lets the whole loop be developed
and demoed before the real one works. A second app, neurover scope, shows, records and compares the
raw electrode signals and runs cued-trial experiments.

```
BrainBit Flex ─┐                                           ┌─► simulated rover
               ├─► eeg/: each device on its own clock ─► SSVEP decode ─► smoother ─► F/B/L/R/S @ 10 Hz ─┤
BrainBit band ─┘    (judge electrodes, weight spikes)                                                 └─► (serial micro:bit later)
```

**The command contract:** every source produces, and every sink consumes, exactly five single ASCII
letters: `F` forward, `B` back, `L` turn left in place, `R` turn right in place, `S` stop. On the wire
to the micro:bit each is the letter plus `\n`, at 115200 baud, resent at 10 Hz even when unchanged.

## Setup (Python 3.10+)

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt        # Windows: .venv\Scripts\pip ...
```

On macOS allow Bluetooth for your terminal app when asked.

## The rover (`python main.py`)

```
python main.py                     # Flex + headband: look at a box to drive
python main.py --devices flex      # just the Flex
python main.py --source mock       # synthetic EEG: arrow keys = which box the virtual driver looks at
python main.py --source keyboard   # WASD / arrows, no EEG: try the sim
```

| flag | meaning |
| --- | --- |
| `--source eeg\|mock\|keyboard` | input (default `eeg`) |
| `--devices LIST` | EEG devices from `config.EEG_DEVICES` (default `flex,headband`) |
| `--user NAME` | whose calibration to load / save (default `default`) |
| `--window S` | SSVEP window, 1-4 s (default: the calibrated one) |
| `--mock-signal clean\|realistic` | mock: strong flicker response (default), or weak + artifacts + reference pops |
| `--config-placement` | ignore the electrode positions set in the scope; use `config.EEG_DEVICES` |
| `--view chase`, `--no-sim`, `--no-log`, `--duration S`, `--screenshot PNG` | |

**Keys:** **V** flicker on/off (a photosensitivity warning first, once per session) · **C** calibrate
· **I** decoder view · **R** measure electrode contact · **P** top-down / chase view · **F5** reset the
rover · **F12** screenshot · **Esc** quit (sends a final S). Mock: arrows = gaze, **J** = jaw clench.

> **Photosensitivity warning:** the flicker (7.5-20 Hz) can trigger seizures in people with
> photosensitive epilepsy. It is off by default and shows a full-screen warning first.

### How a box is decoded (`eeg/ssvep.py`)

Three boxes flicker at 8.57 Hz (↑ F), 20 Hz (← L) and 7.5 Hz (→ R), counted in displayed frames, so
they need a **60 Hz display with vsync** (the app measures it and warns). Looking away at the rover =
rest = S. The frequencies stay clear of the driver's alpha rhythm (10-12 Hz, strongest while looking at
the rover), which made a 10 Hz ↓ box and a 12 Hz ← box drive by themselves (see `config.SSVEP_FRAMES`).

Every 0.1 s, for each streaming device, on its own clock (nothing is merged):

1. its latest window (default 2 s) of 6-40 Hz EEG;
2. **electrodes** (`eeg/clean.py`): each is judged with its device's shared part removed (each
   electrode minus the median of its device-mates). Within one device the electrodes share a large
   common signal (60-97% of the power on the real Flex and headband: that device's reference and
   movement noise), so raw channels go bad together and look "noisy" when they aren't. Only an
   electrode that is flat, saturated, or clearly noisier than its device-mates sits out (after
   ~1.5 s, and comes back ~1.5 s after it looks fine again);
3. **spikes** (blinks, pops) are down-weighted, never a reason to drop the window;
4. **score**: CCA between the electrodes and sine/cosine at each box's frequency and its double,
   compared with the same CCA at nearby frequencies where nothing flickers. That gives a z per box: how
   far it stands out from this device's chance level right now (a noisy device scores ~0, not
   "confident");
5. devices **pool** (Stouffer: z = Σz / √devices): agreeing devices add up, a device with nothing to
   say only dilutes, a device that drops out just leaves. No device at all = no prediction = the
   dead-man stops the rover.

The best box wins if its z clears the threshold and beats the runner-up by the margin; the smoother
then needs N windows in a row before the command changes.

**Moving = stop.** Each device's newest half second is checked against its calm moments: the head
turning (to look at the rover), electrodes shifting (many samples far out at 6-40 Hz) or a clenched
jaw (45-100 Hz muscle 3× calm), on at least half the electrodes, stops the rover at once (no
debounce) and for 1 s after; the dashboard shows **STOP: MOVING** / **STOP: MUSCLE**. Clench to stop.
Sitting still it never fired on a real calibration (thresholds `MOVE_*` in `config.py`).

**Calibrate (C)**, ~4.5 min, done as when driving the real rover: low beep, look at the rover (that's
rest); middle beep, turn back: an arrow points at the next box, 3-2-1; high beep, look at that box
until its yellow frame goes; then back to the rover. Random order, 6 per box, 2 practice trials
first. The recording is saved to `recordings/` (the scope can play it back) and replayed, stop gate
and all, through the same smoother for every window length × threshold × margin × debounce (the
3-2-1 hand-over isn't scored); the fastest setting with < 2% wrong drive commands that keeps still
while you look at the rover wins and is saved per user (`models/ssvep_<user>.json`). Nothing else is
learned. Calibrating again adds up: up to 5 calibrations with the same electrodes and the same rest
(rover, or the older centre cross) are fitted together. A calibration made with other electrodes
isn't loaded. If no setting reaches half the boxes, it says so and keeps the old one. Uncalibrated defaults never drove on their own on the mock, but a weak
response needs calibrating to drive at all.

`python tools/ssvep_eval.py recordings/<calibration>.npz` replays calibrations offline (older
`models/ssvep_*_cal_*.npz` files too): how far the boxes stood out, and what calibration would
choose. Use it to compare electrode setups.

### Dashboard

Source status, the D-pad with the last command sent, raw vs smoothed prediction, a z bar per box with
the threshold, the electrodes (dot: green = decoding, red = left out, grey = no data; level in µV with
the device's shared part removed), the decoded spectrum, the last 10 s of commands, and sinks + FPS.
**I** shows, per box, the z on each headset and the pooled score over the last 10 s.

### Safety (every source)

* **Debounce:** a command changes only after the same prediction wins N consecutive windows.
* **Dead-man:** no fresh prediction for 500 ms → `S`.
* **10 Hz resend:** the rover can stop itself if it stops hearing from us (the sim does: 0.5 s).
* **Obstacle stop:** the sim refuses `F` when its ultrasonic sensor sees anything closer than 15 cm.
* Every command sent is logged to `logs/session_<time>.csv`.

## neurover scope (`python scope.py`): raw feeds, recordings, comparisons, cued trials

A separate, minimal app: no rover, no decoder, no flicker. It shows every electrode of every
device as it arrives, records them, compares recordings electrode by electrode, and runs cued
left / right / rest trials to test whether (imagined) hand movement shows up.

```
python scope.py                  # BrainBit Flex + headband (config.EEG_USE_DEVICES)
python scope.py --devices flex   # just the Flex
python scope.py --mock           # synthetic EEG (hold 1-4 = stare at a box, J = jaw)
python scope.py --report recordings/A.npz recordings/B.npz   # the same numbers as text
python scope.py --trials 20      # cued sessions with 20 trials per class (default 15)
```

* **LIVE:** one row per electrode, grouped by device: the trace (1-40 Hz + notch, or **F** for raw
  minus DC), its spectrum over the last 4 s, RMS / typical peak / 60 Hz mains / DC offset, and a
  status (OK, MARGINAL, NOISY, FLAT, SATURATED). **R** measures contact resistance. ↑↓ trace scale,
  ←→ or [ ] time span.
* **Space** starts / stops a recording, then asks for a name (e.g. `flex-at-Oz`, `staring-12Hz`).
  Each one is `recordings/<date>_<time>_<name>.npz` (format: `eeg/recording.py`): every device's
  samples exactly as they arrived (own timestamps, no resampling), plus serials, electrode map and
  the last resistance values. Quitting while recording saves it.
* **COMPARE (Tab):** pick up to 6 recordings (Space or click). Each electrode gets a panel with the
  recordings' spectra overlaid (shared dB scale, alpha band shaded, flicker frequencies marked)
  and bars for one metric (←→): RMS, typical peak, mains, alpha power / share, SNR at each flicker
  frequency, artifact time, DC offset, missing samples, resistance. Electrodes are matched by
  position. **N** renames a recording.
* **PLAYBACK:** **P** (or the ▶ on a recording) plays it through the LIVE view: Space play/pause,
  ←→ ±2 s, Home/End, − / = speed (0.25-4×), click or drag the timeline. Under the timeline, one
  lane per device shows where it had data (red = Bluetooth gap); ticks above it are logged events
  (contact checks, reconnects, cues; hover for details).
* **Moving electrodes:** click an electrode's name for a head map and click where it is now. A spot
  another lead of the same device has swaps the two; the other device's spots are refused; you can
  also type any name. Live, that's where it is from now on (not while recording), remembered in
  `scope_placement.json`, which **the rover uses too** (`--config-placement` ignores it). In
  PLAYBACK it corrects that recording.
* Spectra leave out 2 s segments with an artifact (> 100 µV); **Artifact time** says how much.

### Cued trials: does (imagined) hand movement show?

One 20 s block per condition doesn't work: the electrodes are still settling, so the condition is
confounded with time (a classifier told the two halves of one block apart as well as it told the
conditions apart). Instead, the classes are interleaved and every trial is compared with its own
baseline.

* **T** (LIVE) sets up a session: task **imagine** or **squeeze** (←→; it's saved with the
  recording), trials per class (↑↓, default 15 = 45 trials, about 7 min), and a name. Enter
  starts recording and the window becomes the cue display: get-ready instructions, 3 practice
  trials, then per trial a fixation cross (2 s, the baseline), the cue for 4 s with a beep (**←**
  left hand, **→** right hand, **○** rest), and a 1.5-3 s random pause (blink there). The order is
  random, never 3 of a kind in a row. There's a break every 15 trials (**Space** continues). A device
  that stops streaming is shown at the bottom. At the end it saves on its own and opens the
  analysis. **Esc** ends early and keeps the finished trials; leads can't be moved meanwhile.
  Put the Flex leads at C3 / C4 (and Cz / Fz) first by clicking their names.
* The recording's `meta` gains `trials`: `[{label, baseline, cue, end}]` (seconds on the samples'
  own clock, the same as `t<k>`; practice trials carry `"practice": true` and aren't analysed) and
  `protocol` (task, classes, timings, seed, planned / completed, ended early).
* **TRIALS** (opens after a session; **T** on a recording in COMPARE or PLAYBACK, or its "N trials"
  chip) shows, per electrode and class, the **ERD**: mu (8-13 Hz) and beta (13-30 Hz) power 0.5-4 s
  after the cue against the 2 s before it, in dB (below 0 = desynchronised; expect the right hand
  at C3 and the left at C4). Bars are mean ± SE over clean trials, with * / ** for p < 0.05 / 0.01
  against 0 and a left-vs-right test; hover for the numbers. On the right: one electrode's
  **time course** (−2 to +5 s around the cue, per class; ←→ or click a panel), the
  **lateralisation** C4 − C3 per class, and a **classifier** per device (shrinkage LDA on each
  trial's log band power, stratified K-fold) against the same with 200 shuffled labels, so chance
  and a p-value are measured, not assumed.
* **A** switches the reference: **device average** (default) subtracts each device's mean across
  its own leads (the shared reference noise above); **as recorded** leaves the samples alone. Trials
  with an artifact (> 100 µV at 1-40 Hz, −2 to +4 s, in the chosen reference) or a Bluetooth gap
  are dropped per device, and the counts are shown.
* PLAYBACK marks each cue on the timeline, coloured by class (blue left, orange right, violet rest).
  `--report` prints the same analysis as text (`--reference raw` for as recorded).
* `--mock` has a motor cortex: mu / beta at C3 and C4 (or the first two leads) that weaken about
  2 dB on the side opposite the cued hand, so the whole thing can be tried without a headset.

## BrainBit notes

* `config.EEG_DEVICES` says which device is which (pinned by serial) and where each SDK channel sits;
  clicking names in the scope overrides the positions for both apps.
* Two devices connect one at a time (a Flex v1 stalls if the other is connecting or measuring
  contact), so with two devices contact is measured on request (**R**), one device at a time.
* If a device drops, the other keeps going; it rejoins on its own.
* A first connection on macOS can fail with "Code 108 Sensor unsupported"; it retries by itself.
* `python tools/brainbit_probe.py` prints what a device reports (family, firmware, channels,
  resistance, signal stats) without starting an app.

## Tests

No hardware needed (the BrainBit SDK is faked in `tests/fake_neurosdk`):

```
python -m pytest
```

## Layout

```
main.py, scope.py    the two apps
config.py            every tunable number
commands.py          the F/B/L/R/S contract and wire encoding
eeg/                 shared by both apps: devices.py (live devices + mock, each on its own clock),
                     brainbit.py (SDK thread), mock.py, filters.py, clean.py (noise handling),
                     ssvep.py (decoding + calibration), recording.py (the .npz format)
rover/               source.py (keyboard + EEG sources), controller.py, smoother.py, sim.py, session_log.py
ui/                  the rover window: app, arena / chase views, dashboard, decoder view, stimulus
rawscope/            the scope window: analysis, playback, head map, cues + trials (+ their UI)
tools/               ssvep_eval.py, brainbit_probe.py
attic/               not in the app: the Myo armband (EMG) and alpha mode, kept to bring back later
tests/               pytest suite
```
