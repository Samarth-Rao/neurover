"""CSV log of every command sent: logs/session_<timestamp>.csv."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

import config

HEADER = ("time", "raw_prediction", "confidence", "command", "blocked")


class SessionLog:
    def __init__(self, directory: Path = config.LOGS_DIR, flush_every: int = 10) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = directory / f"session_{stamp}.csv"
        self._file = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        if self._file.tell() == 0:
            self._writer.writerow(HEADER)
        self._flush_every = max(1, flush_every)
        self._pending = 0
        self.rows = 0

    def log(self, raw_prediction: Optional[str], confidence: float, command: str, blocked: bool,
            when: Optional[datetime] = None) -> None:
        if self._file.closed:
            return
        when = when or datetime.now()
        self._writer.writerow((
            when.isoformat(timespec="milliseconds"),
            "" if raw_prediction is None else raw_prediction,
            f"{confidence:.3f}",
            command,
            int(bool(blocked)),
        ))
        self.rows += 1
        self._pending += 1
        if self._pending >= self._flush_every:
            self._file.flush()
            self._pending = 0

    def close(self) -> None:
        if not self._file.closed:
            self._file.flush()
            self._file.close()
