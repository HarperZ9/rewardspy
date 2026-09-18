# Release readiness

This repository is the HarperZ9 public fork of
[`AvAdiii/rewardspy`](https://github.com/AvAdiii/rewardspy). Preserve upstream
attribution and keep release claims bounded to the package behavior verified
here.

## Current release candidate

- Package name: `rewardspy`
- Version: `0.1.0` in `pyproject.toml` and `rewardspy.__version__`
- License: MIT, from `LICENSE`
- Distribution state: no HarperZ9 GitHub tag or release yet, and no PyPI
  project verified for this fork during the documentation pass
- Release boundary: do not tag, publish to PyPI, or publish release artifacts
  until the draft PR and this checklist are reviewed

## Required checks

Run these from a clean checkout:

```bash
python -m pip install -e ".[dev]"
python -m ruff check .
python -m pytest
python -m build
python scripts/release_smoke.py dist/rewardspy-0.1.0-py3-none-any.whl
```

The smoke test installs the built wheel into a temporary virtual environment and
checks only offline synthetic fixtures:

- `rewardspy --help`
- `rewardspy probe` against a local reward function
- false-success control: malformed `probe` target exits non-zero
- `rewardspy summary` and `rewardspy audit` against a healthy JSONL log
- false-success control: collapsed-reward JSONL log makes `audit` exit non-zero
- `rewardspy export` writes component columns to CSV

## Artifact hashes

After `python -m build`, generate local hashes without publishing:

```bash
python - <<'PY'
from hashlib import sha256
from pathlib import Path

for path in sorted(Path("dist").glob("*")):
    if path.is_file():
        print(f"{sha256(path.read_bytes()).hexdigest()}  {path.name}")
PY
```

Keep the generated artifacts local until review approves tagging or publishing.
