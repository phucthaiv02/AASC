from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from .config import bootstrap_sdk_path


def default_run_artifacts_dir(
    attack_path: Path,
    *,
    root: Path = Path("artifacts/runs"),
    now: datetime | None = None,
) -> Path:
    """Build a distinct artifacts directory from the attack filename and start time."""
    attack_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", attack_path.stem).strip("_") or "attack"
    timestamp = (now or datetime.now().astimezone()).strftime("%Y%m%d_%H%M%S_%f")
    return root / f"{attack_name}_{timestamp}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aas-nim",
        description="Validate an AAS attack locally using NVIDIA NIM models.",
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Run scorer-style red-team validation")
    validate.add_argument("--attack", type=Path, default=Path("attack.py"))
    validate.add_argument(
        "--artifacts-dir",
        type=Path,
        help=(
            "Output directory (default: artifacts/runs/"
            "<attack-file>_<YYYYMMDD_HHMMSS_microseconds>)"
        ),
    )
    validate.add_argument(
        "--model", action="append", dest="models",
        help="NIM model ID; repeat for multiple models (defaults to NIM_MODELS)",
    )
    validate.add_argument("--budget-s", type=float)
    validate.add_argument("--env", choices=("gym", "sandbox"), default="gym")

    leaderboard = subparsers.add_parser(
        "leaderboard", help="Rank saved validation runs and generate an HTML table"
    )
    leaderboard.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    leaderboard.add_argument("--output", type=Path, default=Path("artifacts/leaderboard.html"))
    leaderboard.add_argument("--csv", type=Path, dest="csv_output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv(args.env_file)
    bootstrap_sdk_path()

    if args.command == "leaderboard":
        from .leaderboard import discover_rows, render_csv, render_terminal, write_html

        rows = discover_rows(args.artifacts_dir)
        if not rows:
            print(f"No validation summaries found under {args.artifacts_dir}")
            return 1
        print(render_terminal(rows))
        write_html(rows, args.output)
        print(f"\nHTML leaderboard: {args.output.resolve()}")
        if args.csv_output is not None:
            args.csv_output.parent.mkdir(parents=True, exist_ok=True)
            args.csv_output.write_text(render_csv(rows), encoding="utf-8")
            print(f"CSV leaderboard: {args.csv_output.resolve()}")
        return 0

    from aicomp_sdk.core.env.api import EnvSelection

    from .config import Settings
    from .runner import validate_models

    settings = Settings.from_env()
    artifacts_dir = args.artifacts_dir or default_run_artifacts_dir(args.attack)
    print(f"Artifacts: {artifacts_dir.resolve()}")
    result = validate_models(
        attack_path=args.attack,
        artifacts_dir=artifacts_dir,
        settings=settings,
        models=tuple(args.models) if args.models else None,
        budget_s=args.budget_s,
        env_selection=EnvSelection(args.env),
    )
    print("\nCombined summary:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
