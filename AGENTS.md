# AGENTS.md

This repository is the HarperZ9 public fork of `AvAdiii/rewardspy`: a Python
package for observing reward functions, detecting reward-hacking signatures,
and exporting reward telemetry for review.

Rules:
- Preserve upstream attribution and avoid private/Telos-only rebranding unless
  it is clearly marked as fork-specific documentation.
- Do not commit `.env`, training secrets, private rollout logs, customer data,
  or generated experiment artifacts.
- Keep examples synthetic and deterministic enough for CI.
- Keep README, USAGE, package metadata, CLI behavior, and changelog aligned.
- Run `python -m pytest` and `ruff check .` before release.
- Treat detector output as debugging evidence, not proof that a training run is
  safe or aligned.
