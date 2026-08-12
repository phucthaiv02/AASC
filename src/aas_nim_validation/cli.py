from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from .config import bootstrap_sdk_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aas-nim",
        description="Validate an AAS attack locally using NVIDIA NIM models.",
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Run scorer-style red-team validation")
    validate.add_argument("--attack", type=Path, default=Path("attack.py"))
    validate.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    validate.add_argument(
        "--model", action="append", dest="models",
        help="NIM model ID; repeat for multiple models (defaults to NIM_MODELS)",
    )
    validate.add_argument("--budget-s", type=float)
    validate.add_argument("--env", choices=("gym", "sandbox"), default="gym")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv(args.env_file)
    bootstrap_sdk_path()

    from aicomp_sdk.core.env.api import EnvSelection

    from .config import Settings
    from .runner import validate_models

    settings = Settings.from_env()
    result = validate_models(
        attack_path=args.attack,
        artifacts_dir=args.artifacts_dir,
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

