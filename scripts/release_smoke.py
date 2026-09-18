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

CLI_TIMEOUT_SECONDS = 30
INSTALL_TIMEOUT_SECONDS = 120


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path, help="Path to a built rewardspy wheel")
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    if not wheel.exists():
        raise SystemExit(f"wheel not found: {wheel}")

    with tempfile.TemporaryDirectory(prefix="rewardspy-release-smoke-") as tmp:
        root = Path(tmp)
        sandbox = root / "sandbox"
        sandbox.mkdir()
        venv_dir = root / ".venv"
        venv.EnvBuilder(with_pip=True).create(venv_dir)
        python = _venv_python(venv_dir)
        rewardspy = _venv_script(venv_dir, "rewardspy")
        env = clean_child_env()

        run(
            [str(python), "-m", "pip", "install", str(wheel)],
            cwd=sandbox,
            env=env,
            timeout=INSTALL_TIMEOUT_SECONDS,
        )
        assert_installed_import_origin(python, sandbox, env)
        assert_parent_pythonpath_shadow_is_ignored(python, root, sandbox)
        run([str(rewardspy), "--help"], cwd=sandbox, env=env, expect=0)

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
        run([str(rewardspy), "probe", "my_reward:reward", "-p", str(cases)], cwd=fixtures, env=env)
        run(
            [str(rewardspy), "probe", "missing_colon", "-p", str(cases)],
            cwd=fixtures,
            env=env,
            expect=2,
        )

        healthy = fixtures / "healthy.jsonl"
        hacky = fixtures / "hacky.jsonl"
        write_log(healthy, healthy_records())
        write_log(hacky, hacky_records())

        run([str(rewardspy), "summary", str(healthy), "--last", "50"], cwd=sandbox, env=env)
        run(
            [str(rewardspy), "audit", str(healthy), "--window", "50"],
            cwd=sandbox,
            env=env,
            expect=0,
        )

        # False-success control: a collapsed reward log must fail audit.
        run([str(rewardspy), "audit", str(hacky), "--window", "50"], cwd=sandbox, env=env, expect=1)

        csv_out = fixtures / "healthy.csv"
        run([str(rewardspy), "export", str(healthy), "-o", str(csv_out)], cwd=sandbox, env=env)
        with csv_out.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 200:
            raise SystemExit(f"expected 200 exported rows, saw {len(rows)}")
        if "component.a" not in rows[0]:
            raise SystemExit("CSV export lost component columns")

    print("release smoke passed")
    return 0


def clean_child_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    env.setdefault("PYTHONUTF8", "1")
    return env


def assert_installed_import_origin(python: Path, cwd: Path, env: dict[str, str]) -> None:
    code = """
import importlib.metadata
import os
import pathlib
import rewardspy
import sysconfig

package_file = pathlib.Path(rewardspy.__file__).resolve()
purelib = pathlib.Path(sysconfig.get_paths()["purelib"]).resolve()
assert "PYTHONHOME" not in os.environ
assert "PYTHONPATH" not in os.environ
assert rewardspy.__version__ == importlib.metadata.version("rewardspy")
if purelib not in package_file.parents:
    raise SystemExit(
        f"rewardspy imported from {package_file}, "
        f"outside venv site-packages {purelib}"
    )
print(package_file)
""".strip()
    proc = run([str(python), "-c", code], cwd=cwd, env=env)
    if not proc.stdout.strip():
        raise SystemExit("installed import-origin check did not report rewardspy.__file__")


def assert_parent_pythonpath_shadow_is_ignored(python: Path, root: Path, cwd: Path) -> None:
    shadow = root / "source-shadow"
    package = shadow / "rewardspy"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "raise RuntimeError('release smoke imported rewardspy from PYTHONPATH shadow')\n",
        encoding="utf-8",
    )
    previous = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = str(shadow)
    try:
        assert_installed_import_origin(python, cwd, clean_child_env())
    finally:
        if previous is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = previous


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
    cwd: Path,
    env: dict[str, str],
    expect: int = 0,
    timeout: int = CLI_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    try:
        proc = subprocess.run(
            argv,
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        print("COMMAND:", " ".join(argv), file=sys.stderr)
        print("CWD:", cwd, file=sys.stderr)
        print("TIMEOUT_SECONDS:", timeout, file=sys.stderr)
        print("STDOUT:\n" + _timeout_output(exc.stdout), file=sys.stderr)
        print("STDERR:\n" + _timeout_output(exc.stderr), file=sys.stderr)
        raise SystemExit(124) from exc
    if proc.returncode != expect:
        print("COMMAND:", " ".join(argv), file=sys.stderr)
        print("CWD:", cwd, file=sys.stderr)
        print("EXPECTED:", expect, "GOT:", proc.returncode, file=sys.stderr)
        print("STDOUT:\n" + proc.stdout, file=sys.stderr)
        print("STDERR:\n" + proc.stderr, file=sys.stderr)
        raise SystemExit(proc.returncode or 1)
    return proc


def _timeout_output(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


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
