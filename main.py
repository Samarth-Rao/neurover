"""neurover: drive a (simulated) rover with SSVEP from a BrainBit Flex + headband.

    python main.py                    # BrainBit Flex + headband: look at a flicker box to drive
    python main.py --devices flex     # just the Flex
    python main.py --source mock      # synthetic EEG (arrow keys = which box the virtual driver looks at)
    python main.py --source keyboard  # WASD / arrows, no EEG (try the sim)
    python main.py --camera           # live camera (e.g. iPhone) of the real rover in the middle; K switches camera
    python main.py --rover [NAME]     # also drive the real rover (micro:bit + Maqueen, Bluetooth; firmware/maqueen.ts)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # importable from any working directory
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

SOURCES = ("eeg", "mock", "keyboard")


def parse_args(argv=None) -> argparse.Namespace:
    import config

    p = argparse.ArgumentParser(prog="neurover", description="Drive the rover with SSVEP.")
    p.add_argument("--source", choices=SOURCES, default="eeg", help="input (default: eeg = the BrainBits)")
    p.add_argument("--devices", metavar="LIST", default=",".join(config.EEG_USE_DEVICES),
                   help=f"EEG devices, comma-separated, from config.EEG_DEVICES "
                        f"(default {','.join(config.EEG_USE_DEVICES)})")
    p.add_argument("--user", default="default", help="whose calibration to load / save (default: default)")
    p.add_argument("--window", type=float, metavar="S", help="SSVEP analysis window, 1-4 s (default: calibrated)")
    p.add_argument("--mock-signal", choices=("clean", "realistic"), default="clean",
                   help="mock: 'clean' drives steadily (default); 'realistic' = weak SSVEP, artifacts, reference pops")
    p.add_argument("--config-placement", action="store_true",
                   help="ignore the electrode positions set in the scope; use config.EEG_DEVICES")
    p.add_argument("--camera", nargs="?", type=int, const=0, metavar="N",
                   help="show live camera N (default 0; K switches) in the middle instead of the simulated arena: "
                        "e.g. an iPhone (Continuity Camera) or OBS Virtual Camera pointed at the real rover")
    p.add_argument("--no-sim", action="store_true", help="don't run the simulated rover (the default with --rover)")
    p.add_argument("--sim", action="store_true", help="with --rover: run the simulated rover too")
    p.add_argument("--rover", nargs="?", const="", metavar="NAME_OR_ADDRESS",
                   help="also drive the real rover over Bluetooth: a micro:bit running firmware/maqueen.ts. NAME = "
                        "part of its name, e.g. tegip from 'BBC micro:bit [tegip]' (default: config.MICROBIT_TARGET, "
                        "'' = the first micro:bit seen)")
    p.add_argument("--view", choices=("top", "chase", "camera"),
                   help="what to show in the middle at the start (P cycles; default: camera with --camera, else top)")
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
        pass


def make_source(args):
    import config
    from eeg.devices import Feeds, load_placement
    from rover.source import EegSource, KeyboardSource

    if args.source == "keyboard":
        return KeyboardSource()
    names = [n.strip() for n in args.devices.split(",") if n.strip()]
    placement = {} if args.config_placement else load_placement()
    feeds = Feeds(names, config.EEG_DEVICES, mock=args.source == "mock", mock_preset=args.mock_signal,
                  placement=placement)
    return EegSource(feeds, user=args.user, window_s=args.window, view="camera" if args.camera is not None else "sim")


def main(argv=None) -> int:
    args = parse_args(argv)
    _windows_dpi_awareness()

    import config
    from rover.controller import Controller
    from rover.session_log import SessionLog
    from rover.sim import SimSink
    from ui.app import App

    try:
        source = make_source(args)
    except ValueError as exc:
        print(f"[neurover] {exc}", file=sys.stderr)
        return 2
    sim = None if args.no_sim or (args.rover is not None and not args.sim) else SimSink()   # real rover: no sim
    sinks = [sim] if sim else []
    if args.rover is not None:
        from rover.microbit import MicrobitSink
        sinks.append(MicrobitSink(args.rover or config.MICROBIT_TARGET))
    logger = None if args.no_log else SessionLog()
    controller = Controller(source, sinks, logger=logger)
    camera = None
    if args.camera is not None:
        from rover.camera import Camera
        camera = Camera(args.camera)
    view = args.view or ("camera" if camera is not None else config.ARENA_VIEW)
    app = App(source, controller, sinks, sim=sim, view=view, camera=camera)
    source.start()
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
        print(f"[neurover] {logger.rows} commands logged to {logger.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
