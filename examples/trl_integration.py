"""Using rewardspy with a TRL-style batch reward function.

TRL trainers call reward functions in batch form and expect a list of floats
back. ``watch_trl`` wraps such a function so every completion is recorded, with
no change to your trainer. You would pass ``reward_func`` straight into
``GRPOTrainer(reward_funcs=[reward_func])``.

This demo has no real TRL or model. It simulates the batch call pattern with a
*flawed* reward that quietly pays for length. The "policy" learns to exploit it
by writing longer and longer answers, and rewardspy catches the length drift.

    python examples/trl_integration.py

Press q to quit.
"""

from __future__ import annotations

import random
import threading
import time

import rewardspy
from rewardspy.integrations import watch_trl

random.seed(9)

BATCH_SIZE = 8
STEPS = 700
STEP_DELAY = 0.04


@watch_trl(name="trl_grpo", max_reward=1.3)
def reward_func(prompts, completions, **kwargs):
    """A TRL-style reward with a verbosity loophole: longer answers score more."""
    scores = []
    for completion in completions:
        text = completion if isinstance(completion, str) else str(completion)
        correctness = 1.0 if "ANSWER" in text else 0.0
        length_bonus = min(0.3, len(text) / 8000)  # the flaw the policy will exploit
        scores.append(correctness + length_bonus)
    return scores


def make_batch(step: int) -> tuple[list[str], list[str]]:
    """Completions grow longer over time as the policy chases the length bonus."""
    base_length = 200 + step * 4
    completions = []
    for _ in range(BATCH_SIZE):
        length = max(50, base_length + random.randint(-60, 60))
        completions.append("ANSWER " + "x" * length)
    return ["solve the problem"] * BATCH_SIZE, completions


def train() -> None:
    for step in range(1, STEPS + 1):
        prompts, completions = make_batch(step)
        reward_func(prompts=prompts, completions=completions)
        time.sleep(STEP_DELAY)


def main() -> None:
    trainer = threading.Thread(target=train, daemon=True)
    trainer.start()
    rewardspy.show(reward_func, interval=0.3)


if __name__ == "__main__":
    main()
