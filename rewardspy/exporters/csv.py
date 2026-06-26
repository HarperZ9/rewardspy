"""CSV batch export.

Reward components are dynamic, so the column set is computed from the union of
component keys across all records before writing. Component columns are prefixed
``component.`` to keep them distinct from the base fields.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

from ..records import RolloutRecord

_BASE_FIELDS = [
    "step",
    "timestamp",
    "call_id",
    "scalar_reward",
    "call_duration_ms",
    "input_length",
    "output_length",
]


def write_csv(records: Sequence[RolloutRecord], path: str | Path) -> Path:
    """Write records to ``path`` as CSV and return the path written."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    component_keys = sorted({key for r in records for key in r.components})
    header = _BASE_FIELDS + [f"component.{key}" for key in component_keys]

    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for r in records:
            row = [
                r.step,
                r.timestamp,
                r.call_id,
                r.scalar_reward,
                r.call_duration_ms,
                r.input_length,
                r.output_length,
            ]
            row.extend(r.components.get(key, "") for key in component_keys)
            writer.writerow(row)

    return out
