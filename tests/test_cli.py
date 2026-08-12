from datetime import datetime, timezone
from pathlib import Path

from aas_nim_validation.cli import build_parser, default_run_artifacts_dir


def test_validate_artifacts_dir_defaults_to_automatic():
    args = build_parser().parse_args(["validate", "--attack", "attacks/example.py"])

    assert args.artifacts_dir is None


def test_default_run_artifacts_dir_uses_attack_name_and_timestamp():
    path = default_run_artifacts_dir(
        Path("attacks/live fill.py"),
        now=datetime(2026, 8, 12, 14, 30, 5, 123456, tzinfo=timezone.utc),
    )

    assert path == Path("artifacts/runs/live_fill_20260812_143005_123456")


def test_validate_artifacts_dir_can_still_be_overridden():
    args = build_parser().parse_args(
        ["validate", "--attack", "attack.py", "--artifacts-dir", "custom/run"]
    )

    assert args.artifacts_dir == Path("custom/run")
