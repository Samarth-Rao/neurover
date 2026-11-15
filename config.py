"""Every tunable number for neurorover lives here.

Units are SI (metres, seconds, degrees for angles a human edits) unless a name
says otherwise.
"""
from __future__ import annotations

from pathlib import Path

# --- Paths (relative to this folder, so it works from any working directory) --
ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"
SCREENSHOTS_DIR = ROOT_DIR / "screenshots"

# --- Command stream ----------------------------------------------------------
SEND_HZ = 10.0  # every sink gets the current command at this fixed rate, even if unchanged

# --- Smoother / safety (applies to every source) -----------------------------
SMOOTH_WINDOWS = 4           # same prediction must win this many consecutive windows to switch
CONFIDENCE_THRESHOLD = 0.6   # ...and every one of those windows needs at least this confidence
DEADMAN_S = 0.5              # no fresh data from the source for this long -> force S

# --- Keyboard source -----------------------------------------------------------
KEYBOARD_PERIOD_S = 0.050    # keyboard emits one "window" per hop so smoothing feels identical

# --- Signal pipeline (EMG now, EEG later) ----------------------------------------
WINDOW_S = 0.200             # sliding window length
HOP_S = 0.050                # a new window (and prediction) every hop
KNN_K = 5
EMG_CHANNELS = 8

GESTURES = ("rest", "fist", "spread", "wave_in", "wave_out")
GESTURE_NAMES = {
    "rest": "Rest",
    "fist": "Fist",
    "spread": "Spread",
    "wave_in": "Wave in",
    "wave_out": "Wave out",
}
# Training mode: HOLD one of these number keys to label live windows.
GESTURE_KEYS = {0: "rest", 1: "fist", 2: "spread", 3: "wave_in", 4: "wave_out"}
GESTURE_TO_COMMAND = {
    "rest": "S",
    "fist": "F",
    "spread": "B",
    "wave_in": "L",
    "wave_out": "R",
}

# Myo: the person decides which gesture means which command. One class per command, plus "relax"
# (a resting arm), which also means stop, so a relaxed arm never drives. C runs a guided gesture
# calibration; in training mode (T) hold these number keys while making the gesture.
MYO_CLASSES = ("relax", "forward", "back", "left", "right", "stop")
MYO_CLASS_COMMANDS = {"relax": "S", "forward": "F", "back": "B", "left": "L", "right": "R", "stop": "S"}
MYO_CLASS_NAMES = {"relax": "Relax", "forward": "Forward", "back": "Back", "left": "Left", "right": "Right",
                   "stop": "Stop"}
MYO_LABEL_KEYS = {0: "relax", 1: "forward", 2: "back", 3: "left", 4: "right", 5: "stop"}
# Guided gesture calibration (C): every class held EMG_CAL_HOLD_S, in random order, EMG_CAL_REPS
# times, with a relaxed gap between (~2 min). The first EMG_CAL_SETTLE_S of each hold is skipped.
EMG_CAL_REPS = 3
EMG_CAL_HOLD_S = 4.0
EMG_CAL_GAP_S = 2.0
EMG_CAL_LEAD_S = 5.0
EMG_CAL_SETTLE_S = 0.8
# EMG bars: before calibration each channel scales between its own recent resting level and peak;
# a calibration fixes them to the person's measured relaxed level and strong-gesture level.
EMG_DISPLAY_HISTORY_S = 10.0

# --- Training mode -----------------------------------------------------------------
TRAIN_SETTLE_S = 0.3          # after a label key goes down, ignore windows for this long (get into the pose)
TRAIN_TARGET_PER_CLASS = 200  # windows per gesture the training panel counts as "enough" (~10 s held)
TRAIN_CV_FOLDS = 5            # contiguous (unshuffled) folds for the accuracy estimate shown after fitting

# --- EMG display -----------------------------------------------------------------------
# Myo PREPROCESSED values are unsigned amplitudes; pyomyo's examples plot them on a 0..1024 scale.
EMG_DISPLAY_MIN_FULL_SCALE = 500.0  # bars auto-scale to the recent peak, but never below this
EMG_DISPLAY_PEAK_DECAY = 0.995      # per window

# --- Mock EMG (imitates the Myo's PREPROCESSED stream: 8 rectified, smoothed amplitudes) --
MOCK_SAMPLE_HZ = 50.0
MOCK_TRANSITION_S = 0.12      # time constant of the virtual muscles moving between gestures
MOCK_FULL_SCALE = 800.0       # amplitude at activation 1.0
MOCK_BASELINE = 15.0
MOCK_NOISE = 0.6              # multiplicative, low-pass filtered noise
MOCK_FLOOR_NOISE = 6.0        # additive sensor noise
MOCK_EFFORT_DRIFT = 0.35      # how much overall effort wanders (+/- fraction)
# Arrow keys choose what the virtual arm is doing (none held = rest).
MOCK_ARROW_GESTURES = {"up": "fist", "down": "spread", "left": "wave_in", "right": "wave_out"}
# Activation (0..1) of each of the 8 channels per gesture.
MOCK_PATTERNS = {
    "rest":     (0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03),
    "fist":     (0.55, 0.50, 0.40, 0.35, 0.40, 0.45, 0.55, 0.60),
    "spread":   (0.10, 0.20, 0.60, 0.80, 0.70, 0.25, 0.10, 0.08),
    "wave_in":  (0.70, 0.75, 0.25, 0.08, 0.08, 0.12, 0.45, 0.65),
    "wave_out": (0.08, 0.10, 0.25, 0.45, 0.80, 0.75, 0.35, 0.12),
}

# --- Myo armband --------------------------------------------------------------------
MYO_SCAN_WARN_S = 15.0         # still scanning after this long -> tell the user to wake the armband
MYO_NO_DATA_WARN_S = 1.0       # connected but silent for this long -> show an error (dead-man already stopped us)
MYO_RECONNECT_AFTER_S = 3.0    # connected but silent for this long -> restart the worker and rescan
MYO_RETRY_S = 3.0              # after an error (no dongle, port busy, link lost) retry this often
# How to reach the armband: "dongle" = the Bluegiga USB adapter (pyomyo, 50 Hz preprocessed EMG),
# "ble" = this computer's own Bluetooth (bleak, 200 Hz filtered EMG), "auto" = the dongle if one is
# plugged in, else Bluetooth. The two give differently scaled EMG, so each has its own trained model.
MYO_BACKEND = "auto"
MYO_BLE_NAME = None              # Bluetooth: pick the armband whose name/address contains this (None = first Myo)
MYO_BLE_MIN_FULL_SCALE = 30.0    # Bluetooth EMG is int8: smaller numbers, so the bars need a smaller floor

# --- EEG (BrainBit Flex, and the mock EEG source) -------------------------------------
EEG_SAMPLE_HZ = 250.0
# Each BrainBit is a "device": which kind to look for, and where each of its SDK channels sits on the
# head (SDK channel names in the order the device reports them). Edit "map" to match your placement.
#   flex      BrainBit Flex: 4 movable leads. Plug the lead the SDK calls O1 at PO3, O2 at PO4,
#             T3 at POz and T4 at Pz (or change the map).
#   headband  the classic BrainBit headband: fixed electrodes at O1, O2, T3, T4.
# "serial" pins a device to that unit (as shown when it connects; None = find it by kind). A Flex v1
# and the headband look identical until connected (both "BrainBit"), so pinning saves a handover.
EEG_DEVICES = {
    "flex": {"label": "Flex", "family": "flex", "serial": "450904",
             "map": {"O1": "PO3", "O2": "PO4", "T3": "POz", "T4": "Pz"}},
    "headband": {"label": "Headband", "family": "headband", "serial": "133013",
                 "map": {"O1": "O1", "O2": "O2", "T3": "T3", "T4": "T4"}},
}
EEG_USE_DEVICES = ("flex", "headband")   # which of them to connect (--eeg-devices flex,headband)
# Two devices stream over Bluetooth on their own clocks: their samples are resampled onto one
# 250 Hz timeline, this far behind real time so both devices' packets have arrived.
EEG_MERGE_LAG_S = 0.3
EEG_MERGE_MAX_GAP_S = 0.1        # a device with no samples around a time for this long counts as missing there
EEG_DISPLAY_ORDER = ("O1", "Oz", "O2", "PO3", "POz", "PO4", "P3", "Pz", "P4", "T3", "T4")  # dashboard order
EEG_OCCIPITAL = ("O1", "Oz", "O2", "PO3", "POz", "PO4")  # averaged for the spectrum and alpha mode
# Electrodes used for decoding AND the artifact check (all are still shown). SSVEP lives over the
# occipital / parieto-occipital cortex; one badly seated parietal electrode shouldn't veto every
# window (it did: P3). An electrode the quality check flags NOISY/FLAT/SATURATED is left out live.
EEG_DECODE_POSITIONS = ("O1", "Oz", "O2", "PO3", "POz", "PO4")
# Noise references: extra channels given to each device's CCA (not artifact-checked). Temporal sites
# carry little SSVEP but share that device's reference/muscle noise, which CCA can then subtract.
# Off until the benchmark (tools/ssvep_benchmark.py) shows it helps on your recordings: ("T3", "T4").
EEG_NOISE_REF_POSITIONS: tuple = ()
EEG_NOTCH_HZ = 60.0                               # mains (US); 50.0 in Europe
EEG_DISPLAY_BAND = (1.0, 40.0)                    # traces + artifact check
EEG_ARTIFACT_UV = 100.0                           # any |sample| above this in a window -> drop the window
# SSVEP checks for artifacts in the band it decodes (6-40 Hz): a blink is mostly below 4 Hz and barely
# touches it, so it no longer throws away ~2 s of windows, while jaw/neck muscle and electrode pops
# (broadband) are still caught. False = check 1-40 Hz like alpha mode does.
SSVEP_ARTIFACT_IN_DECODED_BAND = True
# Denoising (SSVEP):
#   blanking   an artifact no longer throws the whole window away: the bad samples (+-EEG_BLANK_PAD_S)
#              are left out and the rest is decoded, as long as SSVEP_MAX_BLANKED or less is missing.
#   muscle     jaw/neck muscle shows as 45-100 Hz activity: samples where an electrode's 45-100 Hz level
#              jumps above EEG_MUSCLE_FACTOR x its own recent baseline (and EEG_MUSCLE_MIN_UV) are
#              blanked for that device. Relative, because the baseline mostly reflects contact quality
#              (4 µV on a good electrode, 20-30 µV on a poor one, measured on real recordings).
#   detrend    leftover slow drift (fresh dry electrodes) is fitted out of each window (polynomial).
SSVEP_MAX_BLANKED = 0.5          # (a 0.05-0.3 s pop blanks ~0.4-0.9 s once filter ringing is included)
EEG_BLANK_PAD_S = 0.05
EEG_MUSCLE_BAND = (45.0, 100.0)
EEG_MUSCLE_FACTOR = 3.0
EEG_MUSCLE_MIN_UV = 8.0
EEG_MUSCLE_BASELINE_S = 20.0
SSVEP_DETREND_ORDER = 2          # 0 = mean only (no detrending)
# Continuous per-electrode signal check (catches bad placement the one-off resistance test misses):
EEG_QUALITY_WINDOW_S = 6.0      # judge each electrode on its last 6 s of windows (median, so blinks don't count)
# Typical window peak (99th pct of |1-40 Hz EEG|, median over 6 s). Dry electrodes on hair often sit
# at 80-150 µV and still decode fine; real contact failures read in the hundreds to thousands.
EEG_QUALITY_WARN_UV = 80.0      # above this = MARGINAL: yellow note, still decoded, calibration allowed
EEG_QUALITY_MAX_UV = 150.0      # above this = NOISY: red, left out of decoding, blocks calibration
# SSVEP judges electrodes in the band it decodes (6-40 Hz): fresh dry electrodes drift by thousands
# of µV at 1-5 Hz while settling, which looks "noisy" at 1-40 Hz but doesn't matter for decoding.
EEG_QUALITY_IN_DECODED_BAND = True
EEG_QUALITY_MIN_UV = 1.0        # below this = FLAT (lead disconnected)
EEG_RAIL_UV = 350_000.0         # raw |signal| this large = amplifier SATURATED (electrode off the skin)
EEG_QUALITY_RECOVER_S = 5.0     # a bad electrode must look good this long before its warning clears
EEG_TRACE_S = 5.0
EEG_TRACE_UV = 50.0                               # trace half-height
EEG_SPECTRUM_HZ = (4.0, 40.0)

# --- BrainBit Flex / headband (--source eeg), via pyneurosdk2 -----------------------------
BRAINBIT_DEVICE = None           # one device only: pick the one whose name/serial contains this text
BRAINBIT_SCAN_WARN_S = 15.0      # still scanning after this long -> show a hint
BRAINBIT_RESIST_S = 4.0          # electrode resistance measurement before streaming (R re-measures)
BRAINBIT_RECONNECT_S = 2.0       # retry interval after a BLE drop (the SDK doesn't reconnect by itself)
BRAINBIT_STALL_S = 3.0           # connected but no samples for this long -> reconnect
EEG_CONTACT_GOOD_OHM = 1_000_000 # BrainBit's recommended maximum electrode resistance (1 MΩ)

# --- SSVEP (--eeg-mode ssvep) ------------------------------------------------------------
SSVEP_REFRESH_HZ = 60.0
# Frames per flicker cycle at 60 Hz: 7 -> 8.57 Hz, 6 -> 10 Hz, 5 -> 12 Hz, 4 -> 15 Hz.
SSVEP_FRAMES = {"up": 7, "down": 6, "left": 5, "right": 4}
SSVEP_COMMANDS = {"up": "F", "down": "B", "left": "L", "right": "R", "rest": "S"}
SSVEP_BAND = (6.0, 40.0)
SSVEP_HARMONICS = 2              # fundamental + 2nd harmonic in the CCA references
SSVEP_WINDOW_S = 2.0             # 1-4 s; longer = more reliable but slower (calibration picks its own)
SSVEP_HOP_S = 0.1                # a new decision every 0.1 s
SSVEP_SMOOTH_WINDOWS = 8         # uncalibrated: the same answer 8 windows running (0.8 s) to switch command
SSVEP_THRESHOLD = 0.47           # defaults (median of mock calibrations) until the user calibrates (C):
SSVEP_MARGIN = 0.09              # one CCA; the best correlation must beat the runner-up by this much
SSVEP_THRESHOLD_FUSED = 0.42     # the same for several devices' scores averaged ("fused" below), which
SSVEP_MARGIN_FUSED = 0.04        # are less spread out, so lower (mock calibrations, 8 seeds)
# With two devices: "fused" = CCA on each device's electrodes, scores averaged (robust to the two
# Bluetooth streams being a few ms out of step); "joint" = one CCA over every decoding electrode.
SSVEP_COMBINE = "fused"
SSVEP_BOX_PX = 220               # flicker squares (design px; never drawn smaller than 200)
SSVEP_REFRESH_TOLERANCE_HZ = 2.0
SSVEP_CAL_SEGMENT_S = 20.0       # (old protocol / alpha-style segments) seconds per segment
SSVEP_CAL_SKIP_S = 2.5           # ignore the start of each segment (window fill + eyes moving)
SSVEP_CAL_MIN_REST_RECALL = 0.9  # calibration keeps at least this share of centre-gaze windows as rest (safety)
# Calibration protocol (C): short cued trials in random order, in blocks with blink breaks between.
# 5 blocks x (4 boxes + 2 centre) x 5 s + breaks = ~2.8 min. More blocks = better fit, longer session.
SSVEP_CAL_BLOCKS = 5
SSVEP_CAL_TRIAL_S = 5.0
SSVEP_CAL_REST_PER_BLOCK = 2     # centre-gaze trials per block (stopping matters as much as going)
SSVEP_CAL_BREAK_S = 4.0
SSVEP_CAL_LEAD_S = 3.0           # "get ready" at the start (not used for fitting)
SSVEP_CAL_SETTLE_S = 0.5         # a training window must start this long after its cue (gaze + brain latency)
# After the session, every decoder x window length x threshold x debounce is replayed through the
# smoother on held-out blocks; the fastest setting whose false commands stay under this share wins.
SSVEP_CAL_WINDOWS = (1.0, 1.5, 2.0, 3.0)
SSVEP_CAL_METHODS = ("pattern", "cca")
SSVEP_CAL_SMOOTH = (1, 2, 3, 5, 8)
SSVEP_CAL_MAX_FALSE = 0.02       # max share of trial time with a wrong drive command
SSVEP_CAL_MAX_REST_DRIVE = 0.01  # max share of (settled) centre-gaze time the rover drives anyway
SSVEP_CAL_MIN_HIT = 0.5          # a calibration that reaches fewer boxes than this is refused (kept the old one)
# Live/diagnostic flicker response: power at a box's frequency (and its double) vs the neighbouring
# frequencies. ~1x = nothing there; a real response is usually 2x or more.
SSVEP_SNR_NEIGHBOURS_HZ = (0.6, 2.0)

# --- Alpha mode (--eeg-mode alpha): eyes closed -> F, eyes open -> S --------------------
ALPHA_BAND = (8.0, 12.0)
ALPHA_TOTAL_BAND = (4.0, 30.0)  # ratio = occipital alpha power / power in this band
ALPHA_WINDOW_S = 2.0
ALPHA_HOP_S = 0.25
ALPHA_THRESHOLD = 0.76          # default (median of mock calibrations) until the user calibrates (C)
ALPHA_CAL_SEGMENT_S = 15.0      # calibration: 15 s eyes open, then 15 s eyes closed
ALPHA_CAL_SKIP_S = 3.0          # ignore the start of each segment (eyes settling + window fill)
ALPHA_CAL_MIN_OPEN_RECALL = 0.95  # eyes open must stay S in at least this share of calibration windows

# --- Mock EEG (synthetic, by physical position; all EEG tests use it) -------------------
MOCK_EEG_NOISE_UV = 10.0         # pink background noise (RMS)
MOCK_EEG_ALPHA_UV = 3.0          # alpha rhythm, eyes open (RMS)
MOCK_EEG_ALPHA_CLOSED_UV = 14.0  # alpha rhythm, eyes closed (hold E)
MOCK_EEG_ALPHA_ATTEND = 0.5      # alpha is suppressed to this fraction while attending a flicker box
MOCK_EEG_ALPHA_BAND = (8.5, 11.5)  # the alpha rhythm's frequency range
MOCK_EEG_SSVEP_UV = 2.4          # SSVEP fundamental amplitude at Oz (2nd harmonic is half)
MOCK_EEG_MAINS_UV = 4.0
MOCK_EEG_DC_UV = 300.0           # electrode offset, removed by the filters
MOCK_EEG_BLINK_EVERY_S = 6.0     # mean interval (0 = off)
MOCK_EEG_JAW_EVERY_S = 30.0      # mean interval (0 = off); hold J to clench on demand
# --source mock-eeg on its own is "clean" by default, for driving it by hand: a strong SSVEP, little
# eyes-open alpha (which the 8.6 and 10 Hz boxes would otherwise pick up while you look at the centre)
# and no random blinks / jaw clenches (J still clenches). --mock-signal realistic uses the values
# above instead: a weak SSVEP and random artifacts, like the tests, to exercise the decoder.
MOCK_EEG_CLEAN = {"ssvep_uv": 6.0, "alpha_uv": 1.0, "artifacts": False}
MOCK_EEG_SPATIAL = {"O1": 0.85, "Oz": 1.0, "O2": 0.85, "PO3": 0.75, "POz": 0.9, "PO4": 0.75,
                    "Pz": 0.55, "P3": 0.5, "P4": 0.5, "T3": 0.15, "T4": 0.15}  # SSVEP/alpha strength by position

# --- Simulated rover -------------------------------------------------------------
ROVER_SPEED_MPS = 0.30        # F / B track speed
ROVER_TURN_DPS = 90.0         # L / R spin-in-place rate
ROVER_ACCEL_MPS2 = 1.0        # per-track acceleration ramp
ROVER_BRAKE_MPS2 = 2.5        # per-track deceleration (S, or reversing)
ROVER_LENGTH_M = 0.14
ROVER_WIDTH_M = 0.11
ROVER_TRACK_WIDTH_M = 0.10    # distance between the two tracks
ROVER_COLLISION_RADIUS_M = 0.09
ROVER_LINK_TIMEOUT_S = 0.5    # like the real rover: stop if no command heard for this long
ROVER_START = (0.5, 1.5, 0.0)  # x m, y m, heading deg (0 = facing +x / screen right, CCW positive)

ULTRASONIC_OFFSET_M = 0.07    # sensor sits this far ahead of the rover centre
ULTRASONIC_MAX_RANGE_M = 2.5
ULTRASONIC_BEAM_DEG = 15.0    # full cone width (HC-SR04 is roughly 15 deg)
ULTRASONIC_RAYS = 5           # rays cast across the cone; the reading is the nearest hit
BLOCK_DISTANCE_M = 0.15       # refuse F when anything is closer than this ahead

ARENA_W_M = 4.0
ARENA_H_M = 3.0
# Axis-aligned boxes (x0, y0, x1, y1) in metres, origin at the bottom-left corner.
ARENA_BOXES = (
    (2.30, 1.25, 2.70, 1.75),   # dead ahead of the start pose: drive F to see BLOCKED
    (1.00, 2.10, 1.50, 2.55),
    (1.10, 0.35, 1.60, 0.75),
    (3.15, 0.45, 3.60, 0.95),
)

SIM_MAX_STEP_S = 1.0 / 240.0  # physics sub-step
SIM_MAX_FRAME_S = 0.25        # never integrate more than this per frame (e.g. after a stall)
TRAIL_S = 20.0                # trail fades out over this many seconds
TRAIL_MIN_STEP_M = 0.01

# --- Chase-cam view (P toggles; --view chase starts in it) -------------------------------
ARENA_VIEW = "top"            # "top" (top-down) or "chase" (behind the rover, rover faces up)
CHASE_PITCH_DEG = 60.0        # camera tilt below horizontal (90 = straight down)
CHASE_FOV_DEG = 40.0          # field of view across the viewport's shorter side (smaller = flatter)
CHASE_DISTANCE_M = 3.3        # camera -> look-at point (bigger = more arena in view)
CHASE_LOOK_AHEAD_M = 0.75     # look-at point is this far ahead of the rover (puts the rover low on screen)
CHASE_YAW_SMOOTH_S = 0.10     # camera low-pass time constants (0 = rigidly locked to the rover)
CHASE_POS_SMOOTH_S = 0.06
CHASE_SNAP_DIST_M = 0.4       # pose jumps bigger than these (F5 reset, teleport) snap the camera
CHASE_SNAP_DEG = 60.0
CHASE_NEAR_M = 0.05           # near clip plane distance
CHASE_WALL_HEIGHT_M = 0.10
CHASE_WALL_THICK_M = 0.05
CHASE_BOX_HEIGHT_M = 0.20
CHASE_TRACK_HEIGHT_M = 0.04
CHASE_BODY_HEIGHT_M = 0.065
CHASE_HALO_RADIUS_M = 0.17
CHASE_TRAIL_WIDTH_M = 0.018
CHASE_TRAIL_BANDS = 24        # trail fade is drawn in this many colour steps (one polygon each)
CHASE_LIGHT_DIR = (-0.6, 0.8)  # world-fixed light (xy, towards the light) used to shade vertical faces


# --- UI ------------------------------------------------------------------------------
WINDOW_SIZE = (1600, 900)
FPS = 60
DASHBOARD_FRACTION = 0.40     # share of the window width given to the dashboard
DASHBOARD_MIN_W = 520
TIMELINE_S = 10.0
