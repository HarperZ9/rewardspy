"""Reward variance collapse.

When every reward in a batch converges to the same value, the model has likely
found a single dominant strategy. If that strategy is a hack, training is now
stuck optimizing it. This detector compares the recent rolling std against a
baseline captured early in the run.
"""

from __future__ import annotations

from ..records import RolloutRecord
from ..store import MetricStore
from .base import BaseDetector, DetectionResult


class VarianceCollapseDetector(BaseDetector):
    name = "variance"
    label = "Variance"

    # (alert_below, warn_below) as a fraction of the baseline std.
    THRESHOLDS = {
        "low": (0.02, 0.10),
        "medium": (0.05, 0.20),
        "high": (0.10, 0.35),
    }

    def __init__(self, sensitivity: str = "medium") -> None:
        super().__init__(sensitivity)
        self._baseline_std: float | None = None

    def check(
        self, store: MetricStore, record: RolloutRecord | None = None
    ) -> DetectionResult:
        if store.window_count < store.window_size:
            return DetectionResult.insufficient_data()

        # Capture the baseline once, from the early part of the run, and freeze it.
        if self._baseline_std is None:
            self._baseline_std = store.baseline_std()

        if self._baseline_std < 1e-6:
            return DetectionResult.ok()

        ratio = store.rolling_std / self._baseline_std
        alert_below, warn_below = self.THRESHOLDS[self.sensitivity]

        if ratio < alert_below:
            return DetectionResult.alert(
                f"Reward variance collapsed to {ratio:.1%} of baseline.",
                "Model may have found a single dominant strategy (possible hack).",
                severity="HIGH",
            )
        if ratio < warn_below:
            return DetectionResult.warning(
                f"Reward variance declining ({ratio:.1%} of baseline).",
                "Watch for convergence onto a single strategy.",
                severity="LOW",
            )
        return DetectionResult.ok()
