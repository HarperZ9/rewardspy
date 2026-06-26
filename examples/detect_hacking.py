"""Watch reward hacking happen, live.

This is a self-contained simulation. There is no real model or GPU: a tiny fake
"policy" generates responses for a math task, and an honest reward function
scores them. The policy learns the task for a while, then discovers a shortcut,
stuffing the answer into an ever longer, formulaic response that maxes the
reward without real reasoning. That is reward hacking.

A background thread runs the "training loop" while the rewardspy dashboard shows
it unfold in real time. Watch the reward curve climb, then the alerts fire:
variance collapse, ceiling saturation, component dominance, and length drift.

Run it:

    python examples/detect_hacking.py

Press q to quit the dashboard.
"""

from __future__ import annotations

import random
import threading
import time

import rewardspy

random.seed(7)

STEPS = 1600
STEP_DELAY = 0.03  # seconds between training steps


def policy(step: int, answer: str) -> str:
    """A fake policy that returns a generated response string.

    Three phases: honest learning, discovering the hack, then full collapse into
    a verbose, always-"correct" template.
    """
    if step < 500:
        # Learning: accuracy climbs from ~0.3 to ~0.8, responses are concise.
        p_correct = 0.3 + 0.5 * (step / 500)
        length = random.randint(500, 900)
    elif step < 850:
        # Discovering the shortcut: nearly always "correct", padding balloons.
        p_correct = min(1.0, 0.8 + (step - 500) / 350 * 0.2)
        length = int(900 + (step - 500) / 350 * 1000)
    else:
        # Collapsed: identical template every time, long but tuned to sit right
        # at the reward ceiling, with no real reasoning.
        p_correct = 1.0
        length = random.randint(1850, 1990)

    correct = random.random() < p_correct
    body = answer if correct else "wrong"
    padding = "a" * max(0, length - 40)
    return f"<think>{body}</think>{padding}"


@rewardspy.watch(
    name="math_grpo_run",
    components=["correctness", "format", "length_pen"],
    max_reward=1.1,
)
def reward(response: str, answer: str) -> dict:
    """An honest reward: correctness + a small format bonus, minus length."""
    correctness = 1.0 if answer in response else 0.0
    format_bonus = 0.1 if response.startswith("<think>") else 0.0
    length_pen = -max(0, len(response) - 2000) / 1000 * 0.05
    total = correctness + format_bonus + length_pen
    return {
        "correctness": correctness,
        "format": format_bonus,
        "length_pen": length_pen,
        "total": total,
    }


def train() -> None:
    for step in range(1, STEPS + 1):
        answer = str(random.randint(0, 99))
        response = policy(step, answer)
        reward(response, answer)
        time.sleep(STEP_DELAY)


def main() -> None:
    trainer = threading.Thread(target=train, daemon=True)
    trainer.start()
    # Launch the live dashboard against the watched reward function.
    rewardspy.show(reward, interval=0.3)


if __name__ == "__main__":
    main()
