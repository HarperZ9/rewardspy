"""Reward slope change via CUSUM.

CUSUM (cumulative sum) is an online change-point method: it accumulates signed
deviations from a baseline and fires when the running sum crosses a threshold. A
sudden, sustained shift in reward-per-step often means the model discovered a
shortcut and switched strategies. CUSUM catches that with a low false-positive
rate, and is simple enough to explain in a README.
"""

from __future__ import annotations

from ..records import RolloutRecord
from ..store import MetricStore
from .base import BaseDetector, DetectionResult


class RewardSlopeChangeDetector(BaseDetector):
    name = "slope"
    label = "Slope"

    # (decision threshold, slack/drift) per sensitivity.
    PARAMS = {
        "low": (8.0, 0.5),
        "medium": (5.0, 0.5),
        "high": (3.0, 0.5),
    }

    # Records to observe before tracking, so the baseline mean is meaningful.
    WARMUP = 20

    def __init__(self, sensitivity: str = "medium") -> None:
        super().__init__(sensitivity)
        self.threshold, self.drift = self.PARAMS[sensitivity]
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0
        self._baseline_mean: float | None = None

    def check(
        self, store: MetricStore, record: RolloutRecord | None = None
    ) -> DetectionResult:
        if record is not None:
            reward = record.scalar_reward
        elif store.records:
            reward = store.records[-1].scalar_reward
        else:
            return DetectionResult.insufficient_data()

        if store.count < self.WARMUP:
            return DetectionResult.insufficient_data()

        if self._baseline_mean is None:
            self._baseline_mean = store.rolling_mean

        deviation = reward - self._baseline_mean
        self.cusum_pos = max(0.0, self.cusum_pos + deviation - self.drift)
        self.cusum_neg = max(0.0, self.cusum_neg - deviation - self.drift)

        if self.cusum_pos > self.threshold or self.cusum_neg > self.threshold:
            direction = "upward" if self.cusum_pos > self.threshold else "downward"
            # Reset and re-baseline so we report each shift once, not every step.
            self.cusum_pos = 0.0
            self.cusum_neg = 0.0
            self._baseline_mean = store.rolling_mean
            return DetectionResult.warning(
                f"Reward distribution shift detected ({direction}).",
                "Policy may have switched strategies. Review recent rollouts.",
                severity="MEDIUM",
            )
        return DetectionResult.ok()
