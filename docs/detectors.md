# How the detectors work

rewardspy runs five independent detectors after every reward call. Each reports
one of: `OK`, `WARNING`, `ALERT`, `INSUFFICIENT_DATA`, or `NOT_APPLICABLE`. The
engine turns actionable results into alerts (once per onset, not once per step)
and the dashboard's Diagnosis panel synthesizes them into a plain-language
verdict.

Sensitivity (`low` / `medium` / `high`) shifts every threshold. `medium` is the
default and the values below.

## Variance collapse

When every reward in a batch converges to the same value, the model has likely
found a single dominant strategy. If that strategy is a hack, training is now
stuck optimizing it.

The detector captures a baseline reward standard deviation early in the run and
compares the recent rolling standard deviation against it:

```
ratio = recent_std / baseline_std
```

`ALERT` when `ratio < 0.05`, `WARNING` when `ratio < 0.20`.

## Reward slope change (CUSUM)

A sudden, sustained shift in reward-per-step often means the model discovered a
shortcut and switched strategies. This uses CUSUM, an online change-point
method that accumulates signed deviations from a baseline:

```
cusum_pos = max(0, cusum_pos + (reward - baseline) - drift)
cusum_neg = max(0, cusum_neg - (reward - baseline) - drift)
```

It fires `WARNING` when either sum crosses the decision threshold (5.0 at medium,
drift 0.5), then resets and re-baselines so each shift is reported once.

## Component dominance

When a reward has several components and one contributes nearly all of the
total, that component is probably being exploited. The classic case is a cheap
format reward dominating over correctness.

```
share = abs(mean[component]) / sum(abs(mean[c]) for c in components)
```

`ALERT` when the top share exceeds 0.90, `WARNING` above 0.75. Requires at least
two components and a minimum sample of records.

## Ceiling saturation

When most rollouts hit the maximum possible reward, the model has saturated the
reward function. That is either genuine mastery or, more often, a ceiling hack.

```
ceiling_rate = fraction of recent rewards >= ceiling * 0.99
```

`ALERT` when `ceiling_rate > 0.80`. The ceiling is the `max_reward` you provide,
or the observed maximum if you do not.

## Response length drift

In LLM RL, length drift is one of the clearest hacking signals: verbosity bias
pushes responses longer, format shortcutting pushes them shorter. The detector
z-scores the recent mean response length against an early baseline:

```
z = (recent_mean_length - baseline_mean_length) / baseline_std_length
```

`WARNING` when `abs(z) > 3.0`.

## GRPO group collapse

Available through `rewardspy.integrations.GRPOSpy`. GRPO normalizes rewards
within a group, so if the within-group variance is zero, every response scored
the same and there is no learning signal. When more than half of recent groups
collapse, GRPOSpy raises an `ALERT` on the same store.
