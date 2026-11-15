"""neurorover: drive a (simulated) rover with body signals.

    python main.py --source myo
    python main.py --source mock
    python main.py --source keyboard
    python main.py --source keyboard --serial COM5
    python main.py --source keyboard --no-sim --serial COM5
"""
from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
from pathlib import Path

# Make the package folders importable no matter where we're launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


def parse_args(argv=None) -> argparse.Namespace:
    from sources import SOURCE_NAMES

    p = argparse.ArgumentParser(prog="neurorover", description="Drive the rover with body signals.")
    p.add_argument("--source", choices=SOURCE_NAMES, default="keyboard", help="input source (default: keyboard)")
    p.add_argument("--serial", metavar="PORT", help="send commands to the micro:bit on this port, e.g. COM5")
    p.add_argument("--myo-port", metavar="PORT", help="Myo Bluegiga dongle port (default: auto-detect)")
    p.add_argument("--myo-backend", choices=("auto", "dongle", "ble"),
                   help="Myo link: Bluegiga dongle or this computer's Bluetooth (default: auto = dongle if plugged in)")
    p.add_argument("--eeg-mode", choices=("ssvep", "alpha"), default="ssvep", help="EEG control mode (default: ssvep)")
    p.add_argument("--ssvep-window", type=float, metavar="S", help="SSVEP analysis window, 1-4 s (default: 2)")
    p.add_argument("--user", default="default", help="whose EEG calibration to load/save (default: default)")
    p.add_argument("--eeg-devices", metavar="LIST",
                   help="EEG devices to use together, comma-separated, from config.EEG_DEVICES "
                        "(default: flex,headband; e.g. --eeg-devices flex)")
    p.add_argument("--eeg-device", metavar="TEXT",
                   help="with one BrainBit: pick it by part of its name or serial (default: strongest)")
    p.add_argument("--mock-signal", choices=("clean", "realistic"), default="clean",
                   help="mock-eeg: 'clean' drives steadily (default); 'realistic' = weak SSVEP + random blinks")
    p.add_argument("--no-sim", action="store_true", help="disable the simulated rover")
    p.add_argument("--view", choices=("top", "chase"), help="arena view to start in (P toggles; default: top)")
    p.add_argument("--no-log", action="store_true", help="don't write logs/session_*.csv")
    p.add_argument("--duration", type=float, metavar="S", help="quit after S seconds (smoke tests)")
    p.add_argument("--screenshot", metavar="PNG", help="save the last frame to this file on exit")
    return p.parse_args(argv)


def _windows_dpi_awareness() -> None:
    """Without this, Windows display scaling blurs the window and can make it larger than the screen."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            import ctypes
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def main(argv=None) -> int:
    args = parse_args(argv)
    _windows_dpi_awareness()

    import config
    from pipeline.controller import Controller
    from pipeline.session_log import SessionLog
    from sinks.sim_sink import SimSink
    from sources import create_source
    from ui.app import App

    options = {}
    if args.source == "myo" and args.myo_port:
        options["port"] = args.myo_port
    if args.source == "myo" and args.myo_backend:
        options["backend"] = args.myo_backend
    if args.source in ("eeg", "mock-eeg"):
        options.update(mode=args.eeg_mode, user=args.user)
        if args.ssvep_window:
            options["window_s"] = args.ssvep_window
        if args.eeg_devices:
            options["devices"] = tuple(d.strip() for d in args.eeg_devices.split(",") if d.strip())
            unknown = [d for d in options["devices"] if d not in config.EEG_DEVICES]
            if unknown or not options["devices"]:
                print(f"[neurorover] unknown --eeg-devices {unknown}; choose from {', '.join(config.EEG_DEVICES)}",
                      file=sys.stderr)
                return 2
    if args.source == "mock-eeg" and args.mock_signal == "clean":
        options.update(config.MOCK_EEG_CLEAN)
    if args.source == "eeg" and args.eeg_device:
        options["device"] = args.eeg_device
    try:
        source = create_source(args.source, **options)
    except (NotImplementedError, ValueError) as exc:
        print(f"[neurorover] {exc}", file=sys.stderr)
        return 2
    sinks = []
    sim = None
    if not args.no_sim:
        sim = SimSink()
        sinks.append(sim)

    if args.serial:
        print("[neurorover] --serial is not implemented yet (milestone 3); ignoring", file=sys.stderr)

    logger = None if args.no_log else SessionLog()
    controller = Controller(source, sinks, logger=logger)
    app = App(source, controller, sinks, sim=sim, view=args.view or config.ARENA_VIEW)

    try:
        source.start()
    except NotImplementedError as exc:
        print(f"[neurorover] {exc}", file=sys.stderr)
        return 2
    try:
        app.run(duration=args.duration, screenshot=args.screenshot)
    except KeyboardInterrupt:
        pass
    finally:
        controller.shutdown()  # final S to every sink
        source.stop()
        for sink in sinks:
            sink.close()
    if logger is not None:
        print(f"[neurorover] {logger.rows} commands logged to {logger.path}")
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()  # the Myo source runs in a child process on Windows
    sys.exit(main())
