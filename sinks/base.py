"""Abstract output sink: anything that consumes the F/B/L/R/S command stream."""
from __future__ import annotations

import abc
from dataclasses import dataclass


@dataclass(frozen=True)
class SinkStatus:
    ok: bool
    label: str      # short name for the dashboard chip, e.g. "SIM" or "SERIAL COM5"
    detail: str = ""


class Sink(abc.ABC):
    name: str = "sink"

    @abc.abstractmethod
    def send(self, command: str) -> None:
        """Deliver one command letter. Must never block the UI thread."""

    def status(self) -> SinkStatus:
        return SinkStatus(True, self.name.upper())

    @property
    def blocked(self) -> bool:
        """True if the sink refused the last command (e.g. F with an obstacle ahead)."""
        return False

    def close(self) -> None:
        pass
