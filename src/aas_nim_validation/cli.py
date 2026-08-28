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
    validate.add_argument(
        "--sequential", action="store_true",
        help="Validate models one at a time instead of concurrently (default: concurrent)",
    )

    validate_gguf = subparsers.add_parser(
        "validate-gguf",
        help=(
            "Run scorer-style red-team validation against the real competition "
            "GGUF weights via llama.cpp on a local GPU (e.g. a rented H100), "
            "instead of the NIM approximation"
        ),
    )
    validate_gguf.add_argument("--attack", type=Path, default=Path("attack.py"))
    validate_gguf.add_argument(
        "--artifacts-dir",
        type=Path,
        help=(
            "Output directory (default: artifacts/runs/"
            "<attack-file>_<YYYYMMDD_HHMMSS_microseconds>)"
        ),
    )
    validate_gguf.add_argument(
        "--model", action="append", dest="models", choices=("gpt_oss", "gemma"),
        help="Repeat for multiple; default: gpt_oss and gemma",
    )
    validate_gguf.add_argument("--budget-s", type=float, default=9000.0)
    validate_gguf.add_argument("--max-tool-hops", type=int, default=8)
    validate_gguf.add_argument("--seed", type=int, default=123)
    validate_gguf.add_argument("--env", choices=("gym", "sandbox"), default="gym")
    validate_gguf.add_argument(
        "--gpt-oss-path", type=Path,
        help="Local gpt-oss .gguf path (else $GPT_OSS_MODEL_PATH, else download from HF)",
    )
    validate_gguf.add_argument(
        "--gemma-path", type=Path,
        help="Local gemma .gguf path (else $GEMMA_MODEL_PATH, else download from HF)",
    )
    validate_gguf.add_argument(
        "--n-ctx", type=int, help="llama.cpp context size (default: the SDK spec's own, 8192)",
    )
    validate_gguf.add_argument(
        "--n-gpu-layers", type=int, default=-1, help="-1 offloads every layer to GPU",
    )
    validate_gguf.add_argument("--n-batch", type=int, default=2048)
    validate_gguf.add_argument("--n-ubatch", type=int, default=1024)
    validate_gguf.add_argument(
        "--no-flash-attn", action="store_true",
        help="Disable flash attention (enable if your llama-cpp-python build lacks FA kernels)",
    )
    validate_gguf.add_argument("--main-gpu", type=int, default=0)
    validate_gguf.add_argument("--n-threads", type=int, help="Default: let llama.cpp auto-detect")

    leaderboard = subparsers.add_parser(
        "leaderboard", help="Rank saved validation runs and generate an HTML table"
    )
    leaderboard.add_argument(
        "--artifacts-dir", type=Path, default=Path("artifacts/validation")
    )
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

    if args.command == "validate-gguf":
        from aicomp_sdk.core.env.api import EnvSelection

        from .gguf_runner import DEFAULT_GGUF_MODELS, GgufPerfConfig, validate_gguf_models

        artifacts_dir = args.artifacts_dir or default_run_artifacts_dir(args.attack)
        print(f"Artifacts: {artifacts_dir.resolve()}")
        model_paths = {}
        if args.gpt_oss_path:
            model_paths["gpt_oss"] = str(args.gpt_oss_path)
        if args.gemma_path:
            model_paths["gemma"] = str(args.gemma_path)
        result = validate_gguf_models(
            attack_path=args.attack,
            artifacts_dir=artifacts_dir,
            models=tuple(args.models) if args.models else DEFAULT_GGUF_MODELS,
            budget_s=args.budget_s,
            max_tool_hops=args.max_tool_hops,
            attack_seed=args.seed,
            env_selection=EnvSelection(args.env),
            perf=GgufPerfConfig(
                n_ctx=args.n_ctx,
                n_gpu_layers=args.n_gpu_layers,
                n_batch=args.n_batch,
                n_ubatch=args.n_ubatch,
                flash_attn=not args.no_flash_attn,
                main_gpu=args.main_gpu,
                n_threads=args.n_threads,
            ),
            model_paths=model_paths,
        )
        print("\nCombined summary:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
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
        parallel=not args.sequential,
    )
    print("\nCombined summary:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
