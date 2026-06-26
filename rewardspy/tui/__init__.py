"""The live terminal dashboard.

``show`` launches the Textual app against a store. The store can be passed
directly, as a watched function, or by name; with a single watched reward
function in the process, it is found automatically.
"""

from __future__ import annotations

from typing import Any

from ..store import MetricStore


def _resolve_store(target: Any) -> MetricStore:
    from ..wrapper import get_store, registered_stores

    if isinstance(target, MetricStore):
        return target
    store = getattr(target, "_rewardspy_store", None) or getattr(target, "store", None)
    if isinstance(store, MetricStore):
        return store
    if isinstance(target, str):
        found = get_store(target)
        if found is None:
            raise ValueError(f"no watched reward function named '{target}'")
        return found

    stores = registered_stores()
    if not stores:
        raise ValueError("no watched reward functions found; wrap one with rewardspy.watch")
    if len(stores) == 1:
        return next(iter(stores.values()))
    raise ValueError(
        "multiple watched functions found; pass one explicitly: "
        f"{', '.join(stores)}"
    )


def show(target: Any = None, *, interval: float = 0.5) -> None:
    """Launch the dashboard for ``target`` (store, watched function, or name)."""
    from .app import RewardSpyApp

    store = _resolve_store(target)
    RewardSpyApp(store, interval=interval).run()


__all__ = ["show"]
