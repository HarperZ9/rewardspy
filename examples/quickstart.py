"""The smallest useful example: wrap a reward function and read the stats.

No dashboard, no training framework. This shows the one-line integration and
that the wrapped function returns exactly what the original returned.

    python examples/quickstart.py
"""

from __future__ import annotations

import rewardspy


@rewardspy.watch(name="quickstart", components=["correctness", "format"])
def reward(response: str, answer: str) -> dict:
    correctness = 1.0 if answer in response else 0.0
    format_bonus = 0.1 if response.startswith("<think>") else 0.0
    return {
        "correctness": correctness,
        "format": format_bonus,
        "total": correctness + format_bonus,
    }


def main() -> None:
    rollouts = [
        ("<think>4+5</think> the answer is 9", "9"),
        ("just 7", "7"),
        ("<think>...</think> probably 3", "5"),
    ]
    for response, answer in rollouts:
        result = reward(response, answer)  # identical to calling it unwrapped
        print(f"answer={answer!r}  reward={result['total']:.2f}")

    store = reward.store
    print()
    print(f"calls recorded : {store.count}")
    print(f"mean reward    : {store.rolling_mean:.3f}")
    print(f"correctness avg: {store.component_stats['correctness'].rolling_mean:.3f}")
    print()
    print("To watch a live run in the terminal, call rewardspy.show(reward).")


if __name__ == "__main__":
    main()
