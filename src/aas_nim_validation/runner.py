from __future__ import annotations

import importlib.util
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from openai import OpenAI

from aicomp_sdk.attacks import AttackRunConfig
from aicomp_sdk.core.env.api import EnvSelection
from aicomp_sdk.evaluation.diagnostics import EvaluatorVerbosity, RunDiagnostics
from aicomp_sdk.evaluation.runner import evaluate_redteam

from .config import Settings
from .nim_agent import NIMChatCompletionsAgent


def load_attack_class(path: Path) -> type[Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Attack file not found: {path}")
    spec = importlib.util.spec_from_file_location("aas_user_attack", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import attack file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    attack_class = getattr(module, "AttackAlgorithm", None)
    if attack_class is None:
        raise RuntimeError(f"{path} must define AttackAlgorithm")
    return attack_class


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "model"


def _write_findings(path: Path, findings: list[Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for finding in findings:
            handle.write(json.dumps(finding, ensure_ascii=False, default=str))
            handle.write("\n")


def _validate_one_model(
    *,
    model: str,
    attack_class: type[Any],
    attack_path: Path,
    artifacts_dir: Path,
    settings: Settings,
    selected_budget: float,
    env_selection: EnvSelection,
    client: OpenAI,
) -> dict[str, Any]:
    label = _slug(model)
    model_dir = artifacts_dir / label
    model_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()

    with RunDiagnostics(
        EvaluatorVerbosity.DEBUG,
        transcript_file=model_dir / "transcript.log",
        event_log_file=model_dir / "framework.jsonl",
        agent_debug_file=model_dir / "agent-debug.jsonl",
    ) as diagnostics:
        debug_sink = diagnostics.make_agent_debug_sink()

        def agent_factory() -> NIMChatCompletionsAgent:
            return NIMChatCompletionsAgent(
                client=client,
                model=model,
                max_tokens=settings.max_tokens,
                temperature=settings.temperature,
                debug_sink=debug_sink,
            )

        execution = evaluate_redteam(
            attack_class,
            budget_s=selected_budget,
            agent_factory=agent_factory,
            agent_label=f"nim:{model}",
            env_selection=env_selection,
            fixtures_dir=None,
            diagnostics=diagnostics,
            attack_env_seed=settings.attack_seed,
            attack_run_config=AttackRunConfig(
                time_budget_s=selected_budget,
                max_tool_hops=settings.max_tool_hops,
            ),
        )

    attack = execution.attack
    if attack is None:
        raise RuntimeError("Red-team evaluation did not return attack results")
    summary = {
        "model": model,
        "score_normalized_0_to_1000": attack.score,
        "score_raw": attack.score_raw,
        "findings_count": attack.findings_count,
        "unique_cells": attack.unique_cells,
        "unique_canonical_cells": attack.unique_canonical_cells,
        "evaluation_time_s": attack.time_taken,
        "wall_time_s": time.time() - started_at,
        "guardrail_id": attack.guardrail_id,
        "guardrail_version": attack.guardrail_version,
        "env": env_selection.value,
        "seed": settings.attack_seed,
        "budget_s": selected_budget,
    }
    (model_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_findings(model_dir / "findings.jsonl", attack.findings)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def validate_models(
    *,
    attack_path: Path,
    artifacts_dir: Path,
    settings: Settings,
    models: tuple[str, ...] | None = None,
    budget_s: float | None = None,
    env_selection: EnvSelection = EnvSelection.GYM,
    parallel: bool = True,
) -> dict[str, Any]:
    attack_class = load_attack_class(attack_path)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    selected_models = models or settings.models
    selected_budget = settings.budget_s if budget_s is None else budget_s
    client = OpenAI(
        base_url=settings.base_url,
        api_key=settings.api_key,
        timeout=settings.timeout_s,
        max_retries=settings.max_retries,
    )

    def run(model: str) -> dict[str, Any]:
        return _validate_one_model(
            model=model,
            attack_class=attack_class,
            attack_path=attack_path,
            artifacts_dir=artifacts_dir,
            settings=settings,
            selected_budget=selected_budget,
            env_selection=env_selection,
            client=client,
        )

    if parallel and len(selected_models) > 1:
        # Independent remote NIM API calls, no shared GPU/model state to contend
        # over, so run every model's full evaluation concurrently in its own
        # thread instead of paying len(selected_models) x budget_s in wall time.
        with ThreadPoolExecutor(max_workers=len(selected_models)) as pool:
            summaries = list(pool.map(run, selected_models))
    else:
        summaries = [run(model) for model in selected_models]

    result = {
        "attack": str(attack_path),
        "models": summaries,
        "local_public_mean": (
            sum(item["score_normalized_0_to_1000"] for item in summaries) / len(summaries)
        ),
    }
    (artifacts_dir / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result
