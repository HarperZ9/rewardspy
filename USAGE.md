# Usage Guide

`rewardspy` wraps reinforcement-learning reward functions so you can observe
scores, components, latency, rollout metadata, and reward-hacking indicators
without changing the reward function's return value.

This repository is a HarperZ9 public fork of
[`AvAdiii/rewardspy`](https://github.com/AvAdiii/rewardspy). Fork-specific
changes should keep upstream behavior intact unless the changelog states the
difference explicitly.

## Install

For local development:

```bash
python -m pip install -e ".[dev]"
```

For package usage:

```bash
python -m pip install rewardspy
```

Optional integrations:

```bash
python -m pip install "rewardspy[trl]"
python -m pip install "rewardspy[wandb]"
python -m pip install "rewardspy[parquet]"
```

## Minimal Python Usage

```python
import rewardspy

def my_reward(response: str, answer: str) -> float:
    return 1.0 if response.strip() == answer else 0.0

watched_reward = rewardspy.watch(my_reward)
score = watched_reward("42", "42")
rewardspy.show(watched_reward)
```

Track reward components by returning a dictionary:

```python
@rewardspy.watch(name="math", components=["correctness", "format", "length"])
def reward(response: str, answer: str) -> dict[str, float]:
    correctness = 1.0 if answer in response else 0.0
    format_score = 0.1 if "<think>" in response else 0.0
    length_penalty = -max(0, len(response) - 2000) / 1000
    return {
        "correctness": correctness,
        "format": format_score,
        "length": length_penalty,
        "total": correctness + format_score + length_penalty,
    }
```

## CLI

Use the CLI against a JSONL reward log:

```bash
rewardspy show logs/run.jsonl --follow
rewardspy summary logs/run.jsonl --last 500
rewardspy audit logs/run.jsonl
rewardspy export logs/run.jsonl -o out.csv
rewardspy probe my_module:reward_fn -p cases.json
```

`audit` exits non-zero when a detector reports a reward-hacking signature. Use
that as a training or CI guard, not as a final safety verdict.

## Examples

```bash
python examples/quickstart.py
python examples/healthy_training.py
python examples/detect_hacking.py
python examples/grpo_math.py
python examples/trl_integration.py
```

The healthy and hacking demos are the fastest way to compare normal reward
behavior with suspicious reward behavior.

## Verification

Before pushing changes in this fork:

```bash
python -m pip install -e ".[dev]"
ruff check .
python -m pytest
```

For public delivery, also run the repository surface sweep used across the
Project Telos public repos when it is available locally:

```bash
python -m public_surface_sweeper . --workspace --json
```

## Boundaries

- `rewardspy` observes reward behavior and detector signatures.
- It does not prove that a model is safe, aligned, or attack-proof.
- It should not publish private training logs, prompts, outputs, credentials,
  customer data, or experiment artifacts.
- Upstream-compatible changes should be small, tested, and suitable for an
  eventual pull request back to `AvAdiii/rewardspy`.
