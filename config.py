"""Every tunable number for neurover lives here.

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

# --- EEG devices (BrainBit Flex + headband, via pyneurosdk2) -------------------------------
EEG_SAMPLE_HZ = 250.0
# Each BrainBit is a "device": which kind to look for, and where each of its SDK channels sits on the
# head (SDK channel names in the order the device reports them). Clicking an electrode's name in the
# scope (python scope.py) overrides "map" for both apps (saved in scope_placement.json).
#   flex      BrainBit Flex: 4 movable leads. Leads 1-4 are SDK channels O1, O2, T3, T4. Now on the back
#             of the head for SSVEP: lead 1 at PO1, lead 2 at POz, lead 3 at PO2, lead 4 at Oz (Oz needs
#             a longer pin; leads 1 and 3 sit up at PO1/PO2, next to POz, so the strap doesn't warp).
#   headband  the classic BrainBit headband: fixed electrodes at O1, O2, T3, T4. Not worn for now (it
#             covers O1/O2, where the Flex is); --devices flex,headband brings it back.
# "serial" pins a device to that unit (as shown when it connects; None = find it by kind). A Flex v1
# and the headband look identical until connected (both "BrainBit"), so pinning saves a handover.
EEG_DEVICES = {
    "flex": {"label": "Flex", "family": "flex", "serial": "450904",
             "map": {"O1": "PO1", "O2": "POz", "T3": "PO2", "T4": "Oz"}},
    "headband": {"label": "Headband", "family": "headband", "serial": "133013",
                 "map": {"O1": "O1", "O2": "O2", "T3": "T3", "T4": "T4"}},
}
EEG_USE_DEVICES = ("flex",)      # which of them to connect (--devices flex,headband)
EEG_DISPLAY_ORDER = ("Fz", "C3", "Cz", "C4", "O1", "Oz", "O2", "PO3", "PO1", "POz", "PO2", "PO4", "P3", "Pz", "P4",
                     "T3", "T4")
EEG_NOTCH_HZ = 60.0                               # mains (US); 50.0 in Europe
EEG_DISPLAY_BAND = (1.0, 40.0)                    # traces
EEG_TRACE_S = 5.0
EEG_TRACE_UV = 50.0                               # trace half-height
EEG_SPECTRUM_HZ = (4.0, 40.0)
EEG_RAIL_UV = 350_000.0          # raw |signal| this large = amplifier SATURATED (electrode off the skin)
EEG_CONTACT_GOOD_OHM = 1_000_000 # BrainBit's recommended maximum electrode resistance (1 MΩ)
BRAINBIT_SCAN_WARN_S = 15.0      # still scanning after this long -> show a hint
BRAINBIT_RESIST_S = 4.0          # electrode resistance measurement (R; also on connect with one device)
BRAINBIT_RECONNECT_S = 2.0       # retry interval after a BLE drop (the SDK doesn't reconnect by itself)
BRAINBIT_STALL_S = 3.0           # connected but no samples for this long -> reconnect

# --- Noise handling (eeg/clean.py) --------------------------------------------------------
# Electrodes are judged on the SSVEP band with their device's shared (reference) part removed. Bad ones
# sit out; nothing ever throws a whole window away (spikes are down-weighted instead).
CLEAN_HISTORY_S = 6.0           # an electrode's typical level = median over this many seconds of windows
CLEAN_HOLD_S = 1.5              # a status change must hold this long (no flicker in and out)
CLEAN_FLAT_UV = 0.3             # typical level below this (as recorded) = FLAT: lead disconnected
CLEAN_NOISY_MAX_UV = 40.0       # typical level above this = NOISY, whatever the others do
CLEAN_NOISY_FACTOR = 3.0        # ...or this many times the device's other electrodes' level
CLEAN_NOISY_FLOOR_UV = 10.0     # ...and above this (never call a quiet electrode noisy)
CLEAN_SPIKE_K = 4.0             # samples beyond this many typical levels are down-weighted, not dropped
# Stop gate: the driver moving (head turning to the rover, electrodes shifting) or clenching the jaw means stop,
# whatever the boxes score. Checked on the newest MOVE_RECENT_S of every device each hop, against the levels of its
# calm moments: "moving" if more than MOVE_SHARE of the samples are beyond CLEAN_SPIKE_K calm levels (6-40 Hz),
# "muscle" if the 45-100 Hz level is above MOVE_MUSCLE_FACTOR x calm; either on at least half the electrodes.
# Sitting still (2026-11-15 calibration) it never fired; a hand-squeezing session, 0.7% of the time.
MOVE_RECENT_S = 0.5
MOVE_SHARE = 0.15
MOVE_MUSCLE_FACTOR = 2.0          # 2x: catches the neck tension of looking at the rover (~70% of it, 2026-11-15)
MOVE_MUSCLE_BAND = (45.0, 100.0)
MOVE_CALM_S = 30.0              # calm levels = median over this long of undisturbed checks
MOVE_RELEARN_S = 10.0           # disturbed this long without a break = the electrodes changed: relearn calm levels
STOP_HOLD_S = 1.0               # stay stopped this long after the disturbance ends

# --- SSVEP (eeg/ssvep.py) -------------------------------------------------------------------
SSVEP_REFRESH_HZ = 60.0
# Frames per flicker cycle at 60 Hz: 7 -> 8.57 Hz, 3 -> 20 Hz, 8 -> 7.5 Hz. Chosen around the driver's alpha rhythm,
# which is strongest at 10-12 Hz while they look at the rover (2026-11-15, rest at the rover): 10 Hz lit up at rest
# (95th percentile z 3.5) and so did 12 Hz (2.5), while 7.5, 8.57, 15 and 20 Hz stayed quiet (1.1-1.4). The 12 Hz ←
# box drove the rover while the driver looked away; the 15 Hz → box responded weakest. No ↓ (B) box: 10 Hz is the
# alpha peak itself. Turning in place covers backing up. (7.5 Hz clashes with a 15 Hz box: twice 7.5 is 15.)
SSVEP_FRAMES = {"up": 7, "left": 3, "right": 8}
SSVEP_COMMANDS = {"up": "F", "left": "L", "right": "R", "rest": "S"}
SSVEP_BAND = (6.0, 40.0)         # decoded band
SSVEP_HARMONICS = 2              # fundamental + 2nd harmonic in the CCA references
SSVEP_HOP_S = 0.1                # a new decision every 0.1 s
# Defaults until the driver calibrates (C). Scores are z: how far a box's frequency stands out from
# frequencies where nothing flickers. These never drove on their own at rest on the mock.
SSVEP_WINDOW_S = 2.0             # 1-4 s; longer = more reliable but slower
SSVEP_Z_THRESHOLD = 5.0
SSVEP_Z_MARGIN = 0.5             # the best box must beat the runner-up by this much
SSVEP_SMOOTH_WINDOWS = 8         # the same answer 8 windows running (0.8 s) to switch command
SSVEP_BOX_PX = 220               # flicker squares (design px; never drawn smaller than 200)
SSVEP_REFRESH_TOLERANCE_HZ = 2.0
# Calibration (C): the scope's cued-trial design, done as when driving the real rover: rest is looking at the rover
# (low beep); middle beep: turn back, an arrow points at the next box while it counts 3-2-1; high beep: look at that
# box (yellow frame) until the frame goes; then back to the rover. The countdown is a hand-over, not fitted, so all
# the rest data is looking at the rover. Random order. 3 boxes x 6 trials x ~13 s + practice + a break = ~4.5 min.
SSVEP_CAL_PER_CLASS = 6          # trials per box
SSVEP_CAL_PRACTICE = 2           # practice trials first (recorded, not used for fitting)
SSVEP_CAL_LEAD_S = 6.0           # instructions + countdown (Space skips)
SSVEP_CAL_AWAY_S = 4.0           # look at the rover, away from the screen (rest)...
SSVEP_CAL_AWAY_JITTER_S = 1.5    # ...4-5.5 s, so the next cue can't be anticipated
SSVEP_CAL_BASELINE_S = 3.0       # turn back: 3-2-1 with an arrow to the next box (not fitted)
SSVEP_CAL_CUE_S = 5.0
SSVEP_CAL_PAUSE_S = (0.0, 0.0)   # no pause: looking at the rover is the rest
SSVEP_CAL_BREAK_EVERY = 9        # trials between breaks
SSVEP_CAL_BREAK_S = 12.0         # (Space skips)
# The recording is replayed through the smoother for every window x threshold x margin x debounce;
# the fastest setting within these limits wins.
SSVEP_CAL_WINDOWS = (1.0, 2.0, 3.0)
SSVEP_CAL_SMOOTH = (1, 2, 3, 5, 8)
SSVEP_CAL_MAX_FALSE = 0.02       # max share of trial time with a wrong drive command
SSVEP_CAL_MAX_REST_DRIVE = 0.01  # max share of (settled) rest time the rover drives anyway
SSVEP_CAL_MIN_HIT = 0.5          # a calibration that reaches fewer boxes than this is refused (kept the old one)
SSVEP_CAL_COMBINE = 5            # fit on the newest calibration + earlier ones with the same electrodes, this many in all

# --- neurover scope numbers (rawscope/analysis.py) -------------------------------------------
EEG_ARTIFACT_UV = 100.0          # a 1-40 Hz sample above this = an artifact (counts as "artifact time")
EEG_QUALITY_WARN_UV = 80.0       # typical peak (99th pct of |1-40 Hz|) above this = MARGINAL
EEG_QUALITY_MAX_UV = 150.0       # ...above this = NOISY
EEG_QUALITY_MIN_UV = 1.0         # ...below this = FLAT
SSVEP_SNR_NEIGHBOURS_HZ = (0.6, 2.0)   # SNR at a box = power there / power this far either side
ALPHA_BAND = (8.0, 12.0)
ALPHA_TOTAL_BAND = (4.0, 30.0)   # "alpha share" = alpha power / power in this band

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


# --- Real rover: micro:bit + Maqueen over Bluetooth LE UART (main.py --rover; rover/microbit.py, firmware/maqueen.ts)
MICROBIT_TARGET = ""             # --rover with no name: "" = the first "BBC micro:bit [.....]" seen. Or part of its name
                                 # ("tegip"; tools/microbit_latency.py --scan lists them) or its address (on macOS a
                                 # CoreBluetooth UUID: macOS never shows the MAC)
MICROBIT_KEEPALIVE_S = 0.2       # resend the current command this often; the firmware stops 0.5 s after the last one
MICROBIT_STALE_S = 0.3           # the app sent nothing for this long (UI stalled / crashed) -> send S
MICROBIT_WRITE_RESPONSE = False  # True: wait for the micro:bit's acknowledgement on every write (the final S always does)
MICROBIT_REQUIRE_PING = True     # drive only once the firmware answers "P" (the old firmware lags seconds behind 10 Hz)
MICROBIT_PING_TIMEOUT_S = 2.0
MICROBIT_SCAN_S = 5.0            # one scan; then MICROBIT_RETRY_S and scan again
MICROBIT_CONNECT_TIMEOUT_S = 10.0
MICROBIT_RETRY_S = 1.0
MICROBIT_WRITE_TIMEOUT_S = 1.0   # a write that takes longer = link dead: reconnect
MICROBIT_CLOSE_S = 3.0           # on quit, wait this long for the final S + disconnect
MICROBIT_TURN_SPEED = 60         # L / R spin speed, 0-255 (forward is 130), sent on connect as "T60"; 0 = the firmware's own.
                                 # At 130 one turn command spun ~180 degrees. Too slow to move at all? Raise it (70-80).
