"""CLI smoke tests for both supported entrypoint invocation styles."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def test_script_entrypoint_help_succeeds() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "src/main.py", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Live breathing-belt acquisition and normalization" in result.stdout

