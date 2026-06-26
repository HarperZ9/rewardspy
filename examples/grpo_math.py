"""A simulated GRPO run, watched with the GRPO-native integration.

GRPO scores a group of responses per prompt. ``GRPOSpy`` watches each group as a
unit. Here the policy learns for a while (groups are diverse, so there is a real
learning signal) and then collapses: every response in a group becomes identical
and correct. When that happens the within-group reward variance hits zero, which
GRPOSpy flags as "no learning signal" on top of the usual hack detectors.

    python examples/grpo_math.py

Press q to quit.
"""

from __future__ import annotations

import random
import threading
import time

import rewardspy
from rewardspy.integrations import GRPOSpy

random.seed(5)

GROUP_SIZE = 8
STEPS = 600
STEP_DELAY = 0.05


def reward_fn(response: str, answer: str) -> dict:
    correctness = 1.0 if answer in response else 0.0
    format_bonus = 0.1 if response.startswith("<think>") else 0.0
    return {
        "correctness": correctness,
        "format": format_bonus,
        "total": correctness + format_bonus,
    }


spy = GRPOSpy(reward_fn, group_size=GROUP_SIZE, name="grpo_math", max_reward=1.1)


def policy_group(step: int, answer: str) -> list[str]:
    """Generate one group of responses for a prompt."""
    learning = step < 250
    p_correct = 0.3 + 0.5 * (step / 250) if learning else 1.0
    group = []
    for _ in range(GROUP_SIZE):
        correct = random.random() < p_correct if learning else True
        body = answer if correct else "wrong"
        group.append(f"<think>{body}</think>")
    return group


def train() -> None:
    for step in range(1, STEPS + 1):
        answer = str(random.randint(0, 99))
        with spy.step(step=step):
            for response in policy_group(step, answer):
                spy.reward(response, answer)
        time.sleep(STEP_DELAY)


def main() -> None:
    trainer = threading.Thread(target=train, daemon=True)
    trainer.start()
    rewardspy.show(spy, interval=0.3)


if __name__ == "__main__":
    main()
