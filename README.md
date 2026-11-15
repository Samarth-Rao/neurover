# neurorover

Drive a small rover with body signals. This app simulates the rover so the
whole control pipeline can be developed and demoed before the physical rover
works.

```
source ──► Prediction ──► smoother ──► F/B/L/R/S @ 10 Hz ──► sinks
(Myo EMG | BrainBit EEG | mock EMG | mock EEG | keyboard)          (sim | serial micro:bit later)
```

**The command contract:** every source produces, and every sink consumes,
exactly five single ASCII letters: `F` forward, `B` back, `L` turn left in
place, `R` turn right in place, `S` stop. On the wire to the micro:bit each one
is the letter followed by `\n`, at 115200 baud, resent at 10 Hz even when
unchanged.

## Setup (Windows 10/11, PowerShell, Python 3.10+)

```powershell
cd neurorover
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If `Activate.ps1` is blocked, run this once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

`pyomyo` from PyPI also pulls in matplotlib and xgboost, which neurorover doesn't
use. To skip them: `pip install --no-deps pyomyo` after installing the other
requirements.

## Run

```powershell
python main.py --source myo        # real Myo armband (Bluegiga dongle plugged in)
python main.py --source mock       # synthetic EMG, no hardware
python main.py --source eeg --eeg-mode alpha   # BrainBit(s): close your eyes to drive
python main.py --source eeg        # BrainBit Flex + headband together: SSVEP flicker control
python main.py --source eeg --eeg-devices flex  # just the Flex
python main.py --source mock-eeg   # synthetic EEG (add --eeg-mode alpha), no hardware
python main.py --source keyboard   # WASD / arrows, no EMG at all
```

| flag | meaning |
| --- | --- |
| `--source {keyboard,mock,myo}` | input source (default `keyboard`) |
| `--myo-port COM7` | Myo dongle port, if auto-detect fails |
| `--eeg-mode ssvep\|alpha` | EEG control: flicker boxes (default) or eyes closed = forward |
| `--eeg-devices LIST` | EEG devices to use together (default `flex,headband`; see `config.EEG_DEVICES`) |
| `--eeg-device TEXT` | with one BrainBit: pick it by part of its name or serial (default: strongest signal) |
| `--ssvep-window S` | SSVEP analysis window, 1–4 s (default 2; try 3 if the rover moves on its own) |
| `--user NAME` | whose EEG calibration to load / save (default `default`) |
| `--view chase` | start in the chase-cam view (P toggles) |
| `--mock-signal realistic` | mock EEG with a weak SSVEP and random artifacts (default `clean`) |
| `--no-sim` | don't run the simulated rover |
| `--no-log` | don't write `logs/session_*.csv` |
| `--duration S` | quit after S seconds (smoke tests) |
| `--screenshot PNG` | save the last frame on exit |

## Driving with the Myo

Two ways to connect (`--myo-backend`, default `auto` = the dongle if one is plugged in):

* **Built-in Bluetooth** (`--myo-backend ble`, no dongle): uses the computer's own Bluetooth via
  `bleak`. Unplug the armband's USB cable (while charging it doesn't stream); if it doesn't show
  up, it's asleep: plug the cable in for a moment and unplug it again, or it's still connected to
  another computer (a Myo talks to one at a time). 200 Hz filtered EMG.
* **Bluegiga dongle** (`--myo-backend dongle`): the USB adapter that came with the Myo, via pyomyo.
  Close **Myo Connect** if it's running (it holds the dongle's port). 50 Hz preprocessed EMG.

The two give differently scaled EMG, so each keeps its own trained model (`models/myo-ble_model.joblib`
vs `models/myo_model.joblib`): train once per backend.

1. Put the armband on and run `python main.py --source myo`. The header goes yellow while it scans,
   then green ("connected via …, battery N%"); the armband vibrates once.
2. **Train** (first time, or after moving the armband):
   * Press **T**. The rover is paused while training (it's sent `S`).
   * For each gesture, **hold its number key while holding the gesture**
     for ~10 s (relax your arm between gestures):
     `0` rest · `1` fist · `2` fingers spread · `3` wave in · `4` wave out.
     The first 0.3 s after pressing a key is skipped so you can settle into the pose.
   * Press **Enter** to fit. The panel shows an accuracy estimate.
   * Press **Ctrl+S** to save (to `models/myo_model.joblib` + `models/myo_data.npz`, or `myo-ble_*` over Bluetooth).
     The saved model loads automatically next time.
   * Press **T** again to drive.
3. Gestures map to commands as: rest→`S`, fist→`F`, spread→`B`,
   wave in→`L`, wave out→`R` (in `config.py`).

Problems show up on screen instead of crashing: no dongle, port busy, armband
not found, or the armband going silent. The rover is stopped (dead-man) and the
Myo worker retries automatically. pyomyo's own output goes to `logs/myo_*.log`.

Mock and Myo keep separate models (`models/mock_*` and `models/myo_*`), so a
model trained on synthetic data never drives the real armband.

## Mock EMG

`--source mock` imitates the Myo's PREPROCESSED stream (8 channels, 50 Hz,
rectified + smoothed). Arrow keys pick what the virtual arm does:
↑ fist, ↓ spread, ← wave in, → wave out, nothing = rest. Train it exactly as
above (e.g. hold `1` and ↑ together).

## BrainBit Flex + headband (`--source eeg`)

Uses BrainBit's `pyneurosdk2` (Windows and macOS; on macOS allow Bluetooth for your
terminal app when asked). By default it connects **two devices at once** and decodes
from all their electrodes (`config.EEG_DEVICES`, `--eeg-devices`):

| device | finds | SDK channel → position |
| --- | --- | --- |
| `flex` | BrainBit Flex (`LEBrainBitFlex`) | O1 → **PO3**, O2 → **PO4**, T3 → **POz**, T4 → **Pz** |
| `headband` | classic BrainBit headband (`LEBrainBit`) | O1, O2, T3, T4 (fixed) |

* **Electrode placement:** plug the Flex lead the SDK calls O1 in at PO3, O2 at PO4,
  T3 at POz and T4 at Pz, or edit that device's `"map"` in `config.EEG_DEVICES`.
* **Decoding** uses the occipital / parieto-occipital electrodes (`EEG_DECODE_POSITIONS`:
  O1, O2 from the headband; PO3, POz, PO4 from the Flex). With `SSVEP_COMBINE = "fused"`
  (default) each device gets its own CCA and the correlations are averaged, which doesn't
  care that the two Bluetooth streams are a few ms out of step; `"joint"` runs one CCA over
  all of them. Pz and T3/T4 are shown but not decoded. T3/T4 can be added as noise
  references (`EEG_NOISE_REF_POSITIONS`) if the benchmark says it helps.
* **One timeline:** the two streams are resampled onto one 250 Hz clock, 0.3 s behind
  real time so both devices' packets have arrived (`EEG_MERGE_LAG_S`).
* **If one device drops** (or isn't on yet), the other keeps driving on its own, with its
  own calibrated threshold; the header says "… (decoding without it)". It rejoins
  automatically once it's back. With no device, no data: the dead-man stops the rover.
* **Bad electrodes** (NOISY / FLAT / SATURATED) are left out of decoding until they've
  looked good for 5 s, instead of vetoing every window; the header stays red to make
  you fix them. An artifact on one device only drops that device from the window.
* On connect the app measures each device's electrode resistance (good < 1 MΩ) and shows
  it under each trace. Poor contacts are named in the header; streaming starts anyway.
  **R** re-measures on both (the rover is stopped meanwhile).
* `python tools/brainbit_probe.py` prints what a headband reports (family,
  firmware, channels, resistance, signal stats) without starting the app.
* A first connection on macOS can fail with "Code 108 Sensor unsupported"; the app
  retries automatically (if it keeps failing, put the headband in pairing mode).

**First test:** wear it, run `python main.py --source eeg --eeg-mode alpha`, wait for
contacts to go green, press **C** for the 30 s alpha calibration (eyes open, then
closed until the beep), then close your eyes: the rover drives; open them: it stops.

## EEG: alpha mode (`--eeg-mode alpha`)

One bit: alpha (8–12 Hz) power on the occipital electrodes (O1, O2, PO3, POz, PO4) divided by total 4–30 Hz power. Eyes
closed → `F`, eyes open → `S`. **C** runs a 30 s calibration (15 s eyes open, 15 s
eyes closed, beeps mark each step) that puts the threshold midway between the two,
saved per user to `models/alpha_<user>.json`. Expect ~2 s to start and ~2 s to stop.

## EEG: SSVEP flicker control

Four boxes flicker around the arena at 8.57 Hz (↑ F), 10 Hz (↓ B), 12 Hz (← L)
and 15 Hz (→ R). Staring at one makes the visual cortex follow that frequency;
CCA on the occipital EEG picks it out. Looking at the rover in the middle = rest = `S`.

> **Photosensitivity warning:** the flicker can trigger seizures in people with
> photosensitive epilepsy. It is off by default and shows a full-screen warning
> (Enter to continue) the first time it is switched on in each session.

* **V** turns the flicker on/off. It needs a **60 Hz** display with vsync: the app
  measures the refresh rate and shows a big warning otherwise (gaming monitors are
  often 144 Hz: Windows Settings › System › Display › Advanced display › 60 Hz).
* **C** runs a guided calibration (~100 s): 20 s looking at the centre, then 20 s on
  each box. It sets this user's threshold and margin, prioritising "the rover never
  drives on its own" (keeps ≥ 90 % of centre-gaze windows as rest), and saves them
  to `models/ssvep_<user>.json`, which loads automatically next time. With two devices
  it also fits each device on its own (used if the other drops out). A calibration made
  with different electrodes isn't loaded; calibrate again after changing the setup.
* Every calibration also saves the raw EEG (`models/ssvep_<user>_cal_<time>.npz`).
  `python tools/ssvep_benchmark.py <a>.npz <b>.npz` compares electrode setups on them
  (each device alone, ± extra electrodes, both joint / fused; CCA and filter-bank CCA),
  trained on one recording and tested on the other; `--methods` compares scoring methods
  and score fusion instead.
* **Denoising** (all in the 6–40 Hz band the decoder uses; settings in `config.py`):
  * Samples above 100 µV (electrode pop, movement) or a **muscle burst** (an electrode's 45–100 Hz
    level jumping above 3× its own recent baseline; jaw/neck tension, loudest at T3/T4) are **patched
    over**: left out of that window, and the rest is decoded. The dashboard shows **PATCHED n%**.
    Patched windows only ever score more cautiously (less data = bigger chance correlations).
  * If more than half a window is bad, that headset sits it out; with no headset left the window is
    dropped (**ARTIFACT**), and no prediction means the dead-man stops the rover.
  * Leftover slow drift (fresh dry electrodes) is fitted out of every window before scoring.
* **Electrode check:** NOISY (red, left out of decoding, blocks calibration) above 150 µV typical
  peak; MARGINAL (yellow, still used) at 80–150 µV; also FLAT and SATURATED. With two headsets, **R**
  measures contact one headset at a time (both pause briefly).
* `--source mock-eeg`: arrows = which box the virtual user stares at,
  **E** = eyes closed (alpha), **J** = jaw clench (artifact). It's "clean" by default
  (strong SSVEP, no random blinks), so holding an arrow drives steadily; add
  `--mock-signal realistic` for a weak SSVEP, resting alpha that leaks into the 8.6 / 10 Hz
  boxes, and random blinks / jaw clenches, which is what the tests use.
* Which SDK channel sits where on the head is set per device in `config.EEG_DEVICES`.

Dashboard (EEG): one trace per electrode labelled by position with contact dots (green =
decoding, grey = shown only or no data, red = bad), the occipital
spectrum with the four stimulus frequencies marked, and the four CCA score bars
with the decision threshold.

## Views

**P** switches between the top-down arena and a chase cam behind the rover (rover
always faces up the screen, slight 3D perspective). Camera settings are `CHASE_*`
in `config.py`.

## Keys

| key | action |
| --- | --- |
| T | training mode on/off (rover paused while on) |
| 0–4 (held, training mode) | label live windows: rest, fist, spread, wave in, wave out |
| Enter | fit the model |
| Ctrl+S | save model + training data to `models/` |
| X, X (training mode) | clear training data (press twice) |
| arrows (mock) | virtual arm gesture |
| WASD / arrows (keyboard source) | drive directly |
| V / C (EEG) | flicker on/off (SSVEP) / guided calibration (Esc cancels) |
| R (BrainBit) | re-measure electrode contact |
| E / J (mock EEG) | eyes closed / jaw clench |
| P | top-down ↔ chase-cam view |
| F5 | reset the simulated rover |
| F12 | save a screenshot to `screenshots/` |
| Esc | quit (sends a final `S`) |

## Safety behaviour (every source)

* **Debounce:** the command only changes after the same prediction wins 4
  consecutive 50 ms windows, each with confidence ≥ 0.6. Otherwise the previous
  command is held. (Windows are 200 ms long and overlap, so 4 windows is ~350 ms
  of signal; if the real armband produces stray commands, raise
  `SMOOTH_WINDOWS` or `CONFIDENCE_THRESHOLD` in `config.py`.)
* **Dead-man:** if the source produces no fresh data for 500 ms, `S` is forced.
* **Fixed 10 Hz resend:** the current command goes to every sink 10 times a
  second, so the real rover can stop itself if it stops hearing from us. The
  simulated rover does exactly that (0.5 s link watchdog).
* **Obstacle stop:** the sim refuses `F` when its ultrasonic sensor sees
  anything closer than 15 cm ahead and shows **BLOCKED**, as the real rover will.

## Dashboard

* source name and connection status (green / yellow = connecting / red)
* D-pad with the active command, plus the letter last sent to the rover
* raw prediction vs. smoothed command, per-class confidence bars, threshold marked
* live EMG channel bars (mock / Myo)
* the last 10 s of commands (raw and sent lanes; blocked sends underlined red)
* sink status and FPS; training panel over the arena in training mode

## Logs

Every sent command is appended to `logs/session_<timestamp>.csv` with
columns `time, raw_prediction, confidence, command, blocked`.

## Tests

No hardware needed (the Myo tests use a fake pyomyo in `tests/fake_pyomyo`):

```powershell
python -m pytest
```

Headless smoke run (no window):

```powershell
$env:SDL_VIDEODRIVER = "dummy"; python main.py --source mock --duration 3 --screenshot smoke.png; Remove-Item Env:SDL_VIDEODRIVER
```

## Layout

```
main.py              CLI entry point
config.py            every tunable number
commands.py          the F/B/L/R/S contract and wire encoding
sources/             base.py (Source), signal_source.py (windowing -> features -> KNN),
                     keyboard, mock EMG, Myo (+ myo_worker.py child process),
                     eeg_common.py (EEG pipeline), mock_eeg_source.py,
                     eeg_source.py + brainbit_worker.py (BrainBits via pyneurosdk2, a thread each),
                     eeg_merge.py (several devices -> one timeline)
pipeline/            features, classifier (+ save/load), eeg (filters/artifacts/spectrum),
                     ssvep (CCA + calibration), alpha, smoother, controller, session log
tools/               brainbit_probe.py (what does the headband report?),
                     ssvep_benchmark.py (compare electrode setups / methods on calibration recordings)
sinks/               simulated rover (sim_sink.py)
ui/                  pygame app, arena view, chase view, dashboard, training panel, SSVEP stimulus
tests/               pytest suite
```
