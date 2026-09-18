"""Offline release smoke test for a built rewardspy wheel.

This script checks the installed CLI in a temporary virtual environment. It
uses synthetic JSONL and probe fixtures only: no provider calls, model downloads,
or training jobs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import venv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path, help="Path to a built rewardspy wheel")
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    if not wheel.exists():
        raise SystemExit(f"wheel not found: {wheel}")

    with tempfile.TemporaryDirectory(prefix="rewardspy-release-smoke-") as tmp:
        root = Path(tmp)
        venv_dir = root / ".venv"
        venv.EnvBuilder(with_pip=True).create(venv_dir)
        python = _venv_python(venv_dir)
        rewardspy = _venv_script(venv_dir, "rewardspy")

        run([str(python), "-m", "pip", "install", "--disable-pip-version-check", str(wheel)])
        run(
            [
                str(python),
                "-c",
                (
                    "import importlib.metadata, rewardspy; "
                    "assert rewardspy.__version__ == importlib.metadata.version('rewardspy')"
                ),
            ]
        )
        run([str(rewardspy), "--help"], expect=0)

        fixtures = root / "fixtures"
        fixtures.mkdir()
        write_reward_module(fixtures)
        cases = fixtures / "cases.json"
        cases.write_text(
            json.dumps(
                [
                    {"response": "the answer is 42", "answer": "42"},
                    {"response": "nope", "answer": "42"},
                ]
            ),
            encoding="utf-8",
        )
        run([str(rewardspy), "probe", "my_reward:reward", "-p", str(cases)], cwd=fixtures)
        run([str(rewardspy), "probe", "missing_colon", "-p", str(cases)], cwd=fixtures, expect=2)

        healthy = fixtures / "healthy.jsonl"
        hacky = fixtures / "hacky.jsonl"
        write_log(healthy, healthy_records())
        write_log(hacky, hacky_records())

        run([str(rewardspy), "summary", str(healthy), "--last", "50"])
        run([str(rewardspy), "audit", str(healthy), "--window", "50"], expect=0)

        # False-success control: a collapsed reward log must fail audit.
        run([str(rewardspy), "audit", str(hacky), "--window", "50"], expect=1)

        csv_out = fixtures / "healthy.csv"
        run([str(rewardspy), "export", str(healthy), "-o", str(csv_out)])
        with csv_out.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 200:
            raise SystemExit(f"expected 200 exported rows, saw {len(rows)}")
        if "component.a" not in rows[0]:
            raise SystemExit("CSV export lost component columns")

    print("release smoke passed")
    return 0


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _venv_script(venv_dir: Path, name: str) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / f"{name}.exe"
    return venv_dir / "bin" / name


def run(
    argv: list[str],
    *,
    cwd: Path | None = None,
    expect: int = 0,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(argv, cwd=cwd, text=True, capture_output=True, check=False)
    if proc.returncode != expect:
        print("COMMAND:", " ".join(argv), file=sys.stderr)
        print("EXPECTED:", expect, "GOT:", proc.returncode, file=sys.stderr)
        print("STDOUT:\n" + proc.stdout, file=sys.stderr)
        print("STDERR:\n" + proc.stderr, file=sys.stderr)
        raise SystemExit(proc.returncode or 1)
    return proc


def write_reward_module(path: Path) -> None:
    (path / "my_reward.py").write_text(
        "def reward(response, answer):\n"
        "    return 1.0 if answer in response else 0.0\n",
        encoding="utf-8",
    )


def record(
    step: int,
    reward: float,
    components: dict[str, float],
    length: int = 10,
) -> dict[str, object]:
    return {
        "call_id": f"c{step}",
        "timestamp": float(step),
        "step": step,
        "scalar_reward": reward,
        "components": components,
        "call_duration_ms": 0.0,
        "input_length": 0,
        "output_length": length,
    }


def healthy_records() -> list[dict[str, object]]:
    return [
        record(
            step,
            (step % 10) / 10.0,
            {"a": 0.25, "b": 0.25, "total": (step % 10) / 10.0},
            length=20 + (step % 7),
        )
        for step in range(200)
    ]


def hacky_records() -> list[dict[str, object]]:
    return [
        record(
            step,
            1.1,
            {"correctness": 1.0, "format": 0.1, "total": 1.1},
            length=10,
        )
        for step in range(120)
    ]


def write_log(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
