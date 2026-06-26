"""Component dominance.

When a reward has several components and one of them contributes nearly all of
the total, that component is probably being exploited. The classic case: a
cheap format reward dominating over the correctness reward.
"""

from __future__ import annotations

from ..records import RolloutRecord
from ..store import MetricStore
from .base import BaseDetector, DetectionResult


class ComponentDominanceDetector(BaseDetector):
    name = "component"
    label = "Component"

    # (alert_above, warn_above) share of total reward held by one component.
    THRESHOLDS = {
        "low": (0.95, 0.85),
        "medium": (0.90, 0.75),
        "high": (0.85, 0.70),
    }

    # Need a reasonable sample before judging dominance, so the first few noisy
    # records do not trigger a false positive.
    MIN_RECORDS = 30

    def check(
        self, store: MetricStore, record: RolloutRecord | None = None
    ) -> DetectionResult:
        if store.count < self.MIN_RECORDS:
            return DetectionResult.insufficient_data()

        means = {
            name: stat.rolling_mean
            for name, stat in store.component_stats.items()
            if name != "total"
        }
        # Dominance only means something with at least two components.
        if len(means) < 2:
            return DetectionResult.not_applicable()

        total = sum(abs(v) for v in means.values())
        if total < 1e-6:
            return DetectionResult.ok()

        shares = {name: abs(value) / total for name, value in means.items()}
        top = max(shares, key=shares.__getitem__)
        share = shares[top]
        alert_above, warn_above = self.THRESHOLDS[self.sensitivity]

        if share > alert_above:
            return DetectionResult.alert(
                f"Component '{top}' dominates reward ({share:.0%}).",
                "Model may be exploiting this component. Check rollouts for shortcut behavior.",
                severity="MEDIUM",
            )
        if share > warn_above:
            return DetectionResult.warning(
                f"Component '{top}' contributes {share:.0%} of reward.",
                "One component is starting to dominate the signal.",
                severity="LOW",
            )
        return DetectionResult.ok()
