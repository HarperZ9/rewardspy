# Reward hacking patterns

A gallery of common reward hacking patterns, what they look like in the data,
and which detector catches them. All of these come from real RL training
failures.

## Ceiling exploitation

The model finds a way to always score the maximum, for example by overwriting
its own unit tests so they always pass, or producing the exact output format
that earns full marks without solving the task.

- Signature: most rollouts at the maximum reward, variance near zero.
- Caught by: **ceiling saturation**, **variance collapse**.
- Try it: `examples/detect_hacking.py`.

## Format or component exploitation

A reward made of several parts can be gamed by maximizing the cheap part. A
format bonus or a length term comes to dominate while correctness stagnates.

- Signature: one component grows to most of the total reward.
- Caught by: **component dominance**.

## Verbosity bias

Many reward models prefer longer answers. The policy learns to pad responses to
collect that preference, regardless of quality.

- Signature: mean response length drifts up sharply.
- Caught by: **response length drift**.
- Try it: `examples/trl_integration.py`.

## Format shortcutting

The mirror image of verbosity: the model emits the minimal correctly-formatted
response that passes a format check, skipping the reasoning.

- Signature: mean response length drifts down sharply.
- Caught by: **response length drift**.

## Strategy switching

The policy discovers a shortcut partway through training and abruptly changes
behavior, visible as a step change in the reward trajectory.

- Signature: a sudden, sustained shift in reward per step.
- Caught by: **reward slope change (CUSUM)**.

## Mode collapse in GRPO

Within a GRPO group, every sampled response becomes effectively identical, so
the group reward variance is zero and the advantages carry no signal.

- Signature: within-group reward variance collapses across many groups.
- Caught by: **GRPO group collapse** (`GRPOSpy`).
- Try it: `examples/grpo_math.py`.

## The common thread

In every case the headline reward keeps rising, which is exactly why these are
missed by watching the reward curve alone. rewardspy contradicts the happy curve
with the statistical signatures above.
