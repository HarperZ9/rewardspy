"""Response length drift.

In LLM RL, length drift is one of the clearest hacking signals: verbosity bias
pushes responses longer, format shortcutting pushes them shorter. This detector
z-scores the recent mean response length against an early baseline.
"""

from __future__ import annotations

import statistics

from ..records import RolloutRecord
from ..store import MetricStore
from .base import BaseDetector, DetectionResult


class LengthDriftDetector(BaseDetector):
    name = "length"
    label = "Length"

    # Absolute z-score above which drift is flagged.
    THRESHOLDS = {"low": 4.0, "medium": 3.0, "high": 2.0}

    def __init__(self, sensitivity: str = "medium") -> None:
        super().__init__(sensitivity)
        self._baseline: tuple[float, float] | None = None

    def check(
        self, store: MetricStore, record: RolloutRecord | None = None
    ) -> DetectionResult:
        window = store.window_size
        if store.count < window * 2:
            return DetectionResult.insufficient_data()

        records = list(store.records)
        if self._baseline is None:
            baseline_lengths = [r.output_length for r in records[:window]]
            self._baseline = (
                statistics.fmean(baseline_lengths),
                statistics.pstdev(baseline_lengths),
            )

        baseline_mean, baseline_std = self._baseline
        if baseline_std < 1.0:
            return DetectionResult.ok()

        recent_mean = statistics.fmean(r.output_length for r in records[-window:])
        z_score = (recent_mean - baseline_mean) / baseline_std
        threshold = self.THRESHOLDS[self.sensitivity]

        if abs(z_score) > threshold:
            direction = "longer" if z_score > 0 else "shorter"
            return DetectionResult.warning(
                f"Response length drifted {direction} by {abs(z_score):.1f} sigma.",
                "Possible verbosity bias or format shortcutting.",
                severity="MEDIUM",
            )
        return DetectionResult.ok()
