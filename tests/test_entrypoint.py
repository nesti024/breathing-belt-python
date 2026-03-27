"""CLI smoke tests for both supported entrypoint invocation styles."""

from __future__ import annotations

import io
from pathlib import Path
import subprocess
import sys

from src.main import prompt_processing_mode


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


def test_prompt_processing_mode_accepts_movement_choice() -> None:
    output = io.StringIO()
    selected_mode, processing_mode = prompt_processing_mode(
        input_func=lambda _: "2",
        output_stream=output,
    )

    assert selected_mode == 2
    assert processing_mode == "movement"


def test_prompt_processing_mode_retries_after_invalid_input() -> None:
    answers = iter(["9", "1"])
    output = io.StringIO()
    selected_mode, processing_mode = prompt_processing_mode(
        input_func=lambda _: next(answers),
        output_stream=output,
    )

    assert selected_mode == 1
    assert processing_mode == "control"
    assert "Invalid selection. Enter 1 or 2." in output.getvalue()


def test_prompt_processing_mode_defaults_to_control_on_empty_or_eof() -> None:
    empty_mode, empty_processing = prompt_processing_mode(
        input_func=lambda _: "",
        output_stream=io.StringIO(),
    )

    def _raise_eof(_: str) -> str:
        raise EOFError

    eof_mode, eof_processing = prompt_processing_mode(
        input_func=_raise_eof,
        output_stream=io.StringIO(),
    )

    assert (empty_mode, empty_processing) == (1, "control")
    assert (eof_mode, eof_processing) == (1, "control")
