"""JSONL streaming sink.

One JSON object per line, appended as each reward call arrives. The file is
flushed after every write so a crashed training run keeps everything recorded
up to the last call.
"""

from __future__ import annotations

import atexit
import json
from collections.abc import Iterable
from pathlib import Path
from typing import IO

from ..records import RolloutRecord

# Track open exporters so they are flushed and closed cleanly at exit.
_OPEN: list[JSONLExporter] = []
_atexit_registered = False


class JSONLExporter:
    """Append-only writer of ``RolloutRecord`` rows to a ``.jsonl`` file."""

    def __init__(self, path: str | Path, *, mode: str = "a") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: IO[str] | None = open(self.path, mode, encoding="utf-8")
        _register_for_cleanup(self)

    def write(self, record: RolloutRecord) -> None:
        if self._fh is None:
            raise ValueError("write to a closed JSONLExporter")
        self._fh.write(json.dumps(record.to_dict(), separators=(",", ":")))
        self._fh.write("\n")
        self._fh.flush()

    def write_many(self, records: Iterable[RolloutRecord]) -> None:
        for record in records:
            self.write(record)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> JSONLExporter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_jsonl(path: str | Path) -> list[RolloutRecord]:
    """Load a ``.jsonl`` file written by :class:`JSONLExporter` back into records."""
    records: list[RolloutRecord] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(RolloutRecord(**json.loads(line)))
    return records


def _register_for_cleanup(exporter: JSONLExporter) -> None:
    global _atexit_registered
    _OPEN.append(exporter)
    if not _atexit_registered:
        atexit.register(_close_all)
        _atexit_registered = True


def _close_all() -> None:
    for exporter in _OPEN:
        exporter.close()
