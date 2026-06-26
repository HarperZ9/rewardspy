"""rewardspy: a plug-in debugger and visualizer for RL reward functions.

One import, zero boilerplate. Wrap a reward function and rewardspy intercepts
every call to track the statistical signatures of reward hacking in real time.

    import rewardspy

    reward_fn = rewardspy.watch(my_reward_fn)
    r = reward_fn(response, ground_truth)   # identical to the original
"""

from .records import Alert, RolloutRecord
from .store import MetricStore
from .wrapper import Session, get_store, registered_stores, watch

__version__ = "0.1.0"

__all__ = [
    "watch",
    "Session",
    "MetricStore",
    "RolloutRecord",
    "Alert",
    "registered_stores",
    "get_store",
    "__version__",
]
