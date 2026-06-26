# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0]

Initial release.

### Added

- `watch` decorator and `Session` for one-line, zero-boilerplate observation of
  reward functions, with component tracking and JSONL export.
- `MetricStore` with O(1) rolling statistics (Welford online algorithm) over a
  bounded sliding window.
- Five reward hacking detectors: variance collapse, reward slope change (CUSUM),
  component dominance, ceiling saturation, and response length drift, with low /
  medium / high sensitivity.
- `DetectionEngine` that raises deduplicated alerts and a synthesized verdict.
- Live Textual dashboard (`rewardspy.show`) with a diagnosis panel, reward curve,
  detector status, component bars, alerts, and recent rollouts.
- Command line interface: `show` (with `--follow`), `summary`, `audit`
  (CI-friendly exit code), and `export`.
- JSONL and CSV exporters.
- Integrations: GRPO-native `GRPOSpy` with group-collapse detection, `watch_trl`
  for TRL batch reward functions, and Weights & Biases logging helpers.
- Examples: `quickstart`, `healthy_training`, `detect_hacking`, `grpo_math`,
  `trl_integration`.
- Documentation: quickstart, API reference, detector explanations, and a reward
  hacking pattern gallery.
