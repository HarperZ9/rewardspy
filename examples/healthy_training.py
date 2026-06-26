"""A healthy training run, for comparison with detect_hacking.py.

Same idea as the hacking demo, but here the reward is well designed and the
model genuinely learns. The reward is a weighted blend of three signals so no
single cheap component can dominate, and the policy improves honestly: it gets
more answers right and writes clearer reasoning, without padding or collapsing
onto a shortcut.

Run this next to the hacking demo to see the contrast. Here the dashboard stays
green and the Diagnosis panel reads "Healthy training". There the same panel
turns red and explains the hack.

    python examples/healthy_training.py

Press q to quit.
"""

from __future__ import annotations

import random
import threading
import time

import rewardspy

random.seed(11)

STEPS = 1600
STEP_DELAY = 0.03


def policy(step: int, answer: str) -> str:
    """An honest policy that slowly gets better at the task."""
    # Accuracy climbs gently from ~0.55 to ~0.75 and then holds, with real noise.
    p_correct = 0.55 + 0.20 * min(1.0, step / 600)
    correct = random.random() < p_correct
    n_steps = random.randint(2, 5)  # reasoning steps shown

    reasoning = "step " * n_steps
    verdict = f"answer={answer}" if correct else "answer=?"
    length = random.randint(400, 700)
    padding = "." * max(0, length - len(reasoning) - len(verdict) - 15)
    return f"<think>{reasoning}{verdict}</think>{padding}"


@rewardspy.watch(name="math_grpo_healthy", components=["answer", "reasoning", "format"])
def reward(response: str, answer: str) -> dict:
    """A balanced, well-designed reward: correctness leads, reasoning supports."""
    answer_ok = 1.0 if f"answer={answer}" in response else 0.0
    reasoning = min(1.0, response.count("step ") / 4.0)
    format_ok = 1.0 if response.startswith("<think>") else 0.0

    answer_term = 0.5 * answer_ok
    reasoning_term = 0.4 * reasoning
    format_term = 0.1 * format_ok
    return {
        "answer": answer_term,
        "reasoning": reasoning_term,
        "format": format_term,
        "total": answer_term + reasoning_term + format_term,
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
    rewardspy.show(reward, interval=0.3)


if __name__ == "__main__":
    main()
