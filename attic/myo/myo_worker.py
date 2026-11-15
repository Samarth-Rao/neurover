"""Child-process side of the Myo source. Keep imports light: this runs in a spawned process.

pyomyo API used here (confirmed from pyomyo 0.0.5's source and the repo's examples,
e.g. examples/myo_multithreading_examp.py and plot_emgs.py):

    from pyomyo import Myo, emg_mode
    m = Myo(tty=None, mode=emg_mode.PREPROCESSED)  # tty=None: auto-detect the dongle (USB PID 2458:0001)
                                                   # raises ValueError('Myo dongle not found!')
    m.add_emg_handler(lambda emg, moving: ...)     # emg: 8 values; PREPROCESSED = 50 Hz, rectified + filtered
    m.add_battery_handler(lambda level: ...)
    m.connect()                                    # scans until an armband answers (no timeout)
    m.set_leds(logo_rgb, bar_rgb); m.vibrate(1)
    while True: m.run()                            # reads one BLE packet, dispatches handlers
    m.disconnect()

Note: ``mode`` must be the enum member. pyomyo's default ``mode=1`` is a plain int,
which never equals ``emg_mode.PREPROCESSED``, so no EMG would be streamed.

Messages put on the queue (all tuples):
    ("scanning", port)        dongle opened, looking for the armband
    ("connected", port)
    ("emg", t, (v0..v7))      t = time.monotonic() in this process
    ("battery", percent)
    ("error", message)        the worker is about to exit
"""
from __future__ import annotations

import os
import sys
import time

LED_COLOR = [0, 160, 255]


def run(q, stop, port, log_path) -> None:
    # pyomyo prints a line for every BLE scan response; keep that out of the console.
    try:
        sink = open(log_path, "a", buffering=1, encoding="utf-8") if log_path else open(os.devnull, "w")
        sys.stdout = sys.stderr = sink
    except OSError:
        pass

    def put(*msg) -> None:
        try:
            q.put(msg)
        except Exception:
            pass

    try:
        from pyomyo import Myo, emg_mode
    except ImportError as exc:
        put("error", f"pyomyo is not installed ({exc}). Run: pip install pyomyo")
        return

    try:
        m = Myo(tty=port, mode=emg_mode.PREPROCESSED)
    except ValueError:
        put("error", "Myo dongle not found. Plug in the Bluegiga USB dongle (or pass --myo-port COMx).")
        return
    except Exception as exc:  # serial.SerialException: wrong port, or it's in use
        put("error", f"Can't open the Myo dongle ({exc}). Close Myo Connect or anything else using it.")
        return

    try:
        dongle = m.bt.ser.port
    except Exception:
        dongle = port or "dongle"
    put("scanning", dongle)

    m.add_emg_handler(lambda emg, moving: put("emg", time.monotonic(), tuple(int(v) for v in emg)))
    m.add_battery_handler(lambda level: put("battery", int(level)))
    try:
        m.connect()
        put("connected", dongle)
        m.set_leds(LED_COLOR, LED_COLOR)
        m.vibrate(1)
        while not stop.is_set():
            m.run()
    except Exception as exc:
        put("error", f"Myo connection lost ({exc}).")
    finally:
        try:
            m.disconnect()
        except Exception:
            pass
