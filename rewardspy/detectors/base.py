"""Detector contract and the result type they return.

Every detector is independent and answers one question about the current state
of a store: does this look like reward hacking? Each returns a
``DetectionResult`` with one of five statuses, so the engine and the dashboard
can treat them uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from ..records import RolloutRecord
from ..store import MetricStore


class Status(str, Enum):
    OK = "OK"
    WARNING = "WARNING"
    ALERT = "ALERT"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(slots=True)
class DetectionResult:
    status: Status
    message: str = ""
    detail: str = ""
    severity: str = ""

    @property
    def actionable(self) -> bool:
        """True when this result should surface as an alert."""
        return self.status in (Status.WARNING, Status.ALERT)

    @classmethod
    def ok(cls) -> DetectionResult:
        return cls(Status.OK)

    @classmethod
    def warning(cls, message: str, detail: str = "", severity: str = "LOW") -> DetectionResult:
        return cls(Status.WARNING, message, detail, severity)

    @classmethod
    def alert(cls, message: str, detail: str = "", severity: str = "HIGH") -> DetectionResult:
        return cls(Status.ALERT, message, detail, severity)

    @classmethod
    def insufficient_data(cls) -> DetectionResult:
        return cls(Status.INSUFFICIENT_DATA)

    @classmethod
    def not_applicable(cls) -> DetectionResult:
        return cls(Status.NOT_APPLICABLE)


class BaseDetector(ABC):
    """Base class for all detectors.

    ``name`` is a stable identifier used in alerts and JSON. ``label`` is the
    short text shown in the dashboard's hack-status panel.
    """

    name: str = "detector"
    label: str = "Detector"

    def __init__(self, sensitivity: str = "medium") -> None:
        if sensitivity not in ("low", "medium", "high"):
            raise ValueError("sensitivity must be 'low', 'medium', or 'high'")
        self.sensitivity = sensitivity

    @abstractmethod
    def check(self, store: MetricStore, record: RolloutRecord | None = None) -> DetectionResult:
        """Inspect the store (and the latest record) and report a status."""
        raise NotImplementedError
