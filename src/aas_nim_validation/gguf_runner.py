from __future__ import annotations

import dataclasses
import gc
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from aicomp_sdk.attacks import AttackRunConfig
from aicomp_sdk.core.env.api import EnvSelection
from aicomp_sdk.evaluation.diagnostics import EvaluatorVerbosity, RunDiagnostics
from aicomp_sdk.evaluation.runner import evaluate_redteam

from .runner import _slug, _write_findings, load_attack_class

# The competition's own notebook targets Kaggle's 2xT4 accelerator. Running the
# same GGUF weights through llama.cpp on one fast datacenter GPU (e.g. H100) is
# both faster (more compute, more VRAM headroom for bigger batches) and a truer
# timing proxy than the NIM-hosted approximation: NIM's per-candidate latency
# profile does not match the real GGUF-on-GPU backend (observed directly: NIM
# round-trips average ~1-4s while the competition's own docstrings put the real
# backend at ~8.5-20s/candidate), which breaks any timing-sensitive tuning
# (MARGIN_S, SPLIT_THRESHOLD_S, etc.) validated only against NIM.
DEFAULT_GGUF_MODELS: tuple[str, ...] = ("gpt_oss", "gemma")


@dataclasses.dataclass(frozen=True)
class GgufPerfConfig:
    """llama.cpp performance knobs for a single fast GPU.

    ``GgufModelServer`` (from the competition SDK) only forwards model_path,
    n_ctx, n_gpu_layers, and verbose to ``Llama(...)`` -- everything else here is
    injected via a wrapped ``llama_cls`` (see ``_tuned_llama_cls``). Defaults
    assume one GPU with plenty of VRAM headroom beyond what a 20B/26B Q4_K_M GGUF
    needs (~12-16GB): bigger batches and flash attention cut wall-clock per
    candidate, which is what actually limits how many validation-fill candidates
    a fixed local budget can afford to try.
    """

    n_ctx: int | None = None  # None -> keep the SDK spec's own default (8192)
    n_gpu_layers: int = -1  # -1 == offload every layer to GPU
    n_batch: int = 2048
    n_ubatch: int = 1024
    flash_attn: bool = True
    offload_kqv: bool = True
    main_gpu: int = 0
    n_threads: int | None = None  # None -> let llama.cpp auto-detect

    def llama_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "n_batch": self.n_batch,
            "n_ubatch": self.n_ubatch,
            "flash_attn": self.flash_attn,
            "offload_kqv": self.offload_kqv,
            "main_gpu": self.main_gpu,
        }
        if self.n_threads is not None:
            kwargs["n_threads"] = self.n_threads
            kwargs["n_threads_batch"] = self.n_threads
        return kwargs


def _model_spec(model_name: str) -> Any:
    if model_name == "gpt_oss":
        from kaggle_evaluation.jed_attack_134815 import gpt_oss_model_server as mod
    elif model_name == "gemma":
        from kaggle_evaluation.jed_attack_134815 import gemma_model_server as mod
    else:
        raise ValueError(f"Unknown GGUF model: {model_name!r} (expected 'gpt_oss' or 'gemma')")
    return mod.SPEC


def _tuned_llama_cls(perf: GgufPerfConfig) -> Callable[..., Any]:
    """Wrap ``llama_cpp.Llama`` with the perf kwargs baked in.

    ``GgufModelServer._load_backend`` always calls
    ``llama_cls(model_path=..., n_ctx=..., n_gpu_layers=..., verbose=...)`` with no
    way to add extra kwargs, so this closure supplies the rest (n_batch,
    flash_attn, ...) while letting the explicit call-site kwargs win on overlap.
    """
    from llama_cpp import Llama

    extra = perf.llama_kwargs()

    def _construct(**kwargs: Any) -> Any:
        return Llama(**{**extra, **kwargs})

    return _construct


def build_gguf_agent_factory(
    model_name: str,
    *,
    perf: GgufPerfConfig,
    model_path: str | None = None,
    llama_cls: Callable[..., Any] | None = None,
) -> tuple[Callable[[], Any], Any]:
    """Load a GGUF model server for ``model_name`` ('gpt_oss' or 'gemma').

    Returns ``(agent_factory, server)``. The caller must call
    ``unload_gguf_server(server)`` when done with this model to free GPU memory
    before loading the next one (models are evaluated sequentially, matching how
    the real grader only ever has one model loaded at a time).
    """
    from kaggle_evaluation.jed_attack_134815.gguf_model_server import GgufModelServer

    spec = _model_spec(model_name)
    if perf.n_ctx is not None:
        spec = dataclasses.replace(spec, n_ctx=perf.n_ctx)
    spec = dataclasses.replace(spec, n_gpu_layers=perf.n_gpu_layers)

    if model_path:
        os.environ[spec.model_path_env_var] = model_path

    resolved_llama_cls = llama_cls if llama_cls is not None else _tuned_llama_cls(perf)
    server = GgufModelServer(spec, llama_cls=resolved_llama_cls)
    t0 = time.monotonic()
    server.load_model()
    print(f"Loaded {model_name} GGUF in {time.monotonic() - t0:.1f}s")
    return (lambda: server._load_agent()), server


def unload_gguf_server(server: Any, *, label: str = "model") -> None:
    """Free GPU memory before loading the next model (mirrors the notebook)."""
    try:
        server.unload()
    except Exception as err:  # defensive: unload must never abort a validation run
        print(f"{label} unload error: {err!r}")
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def validate_gguf_models(
    *,
    attack_path: Path,
    artifacts_dir: Path,
    models: tuple[str, ...] = DEFAULT_GGUF_MODELS,
    budget_s: float = 9000.0,
    max_tool_hops: int = 8,
    attack_seed: int = 123,
    env_selection: EnvSelection = EnvSelection.GYM,
    fixtures_dir: Path | None = None,
    perf: GgufPerfConfig | None = None,
    model_paths: dict[str, str] | None = None,
    llama_cls: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """GGUF/llama.cpp counterpart of ``runner.validate_models`` (the NIM path).

    Writes the same ``artifacts/runs/<attack>_<ts>/<model>/summary.json`` shape so
    ``aas-nim leaderboard`` treats NIM and GGUF runs uniformly, but drives the
    actual competition GGUF weights locally instead of a hosted NIM model --
    giving realistic per-candidate timing on hardware faster than the
    competition's own 2xT4 baseline.
    """
    perf = perf or GgufPerfConfig()
    model_paths = model_paths or {}
    attack_class = load_attack_class(attack_path)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for model in models:
        label = _slug(model)
        model_dir = artifacts_dir / label
        model_dir.mkdir(parents=True, exist_ok=True)
        started_at = time.time()

        agent_factory, server = build_gguf_agent_factory(
            model,
            perf=perf,
            model_path=model_paths.get(model),
            llama_cls=llama_cls,
        )
        # ``evaluate_redteam`` raises a bare, UNCAUGHT ``TimeoutError`` if
        # either the attack-generation or the candidate-replay phase overruns
        # its own budget window (each phase gets its own separate budget_s --
        # a fast generation phase can hand replay far more candidates than
        # replay's OWN window can get through in time). Unlike the real
        # competition gateway (which degrades gracefully, keeping whatever
        # replay finished before its deadline), this SDK path has no such
        # fallback: left unhandled here it would abort the WHOLE sequential
        # loop, losing every model not yet evaluated (and, worse on a rented
        # GPU, skip the ``finally`` unload if not careful). Catching it means
        # one model's timeout is recorded as a zero-score, clearly-errored
        # row and the loop still moves on to the next model.
        try:
            try:
                with RunDiagnostics(
                    EvaluatorVerbosity.DEBUG,
                    transcript_file=model_dir / "transcript.log",
                    event_log_file=model_dir / "framework.jsonl",
                    agent_debug_file=model_dir / "agent-debug.jsonl",
                ) as diagnostics:
                    execution = evaluate_redteam(
                        attack_class,
                        budget_s=budget_s,
                        agent_factory=agent_factory,
                        agent_label=f"gguf:{model}",
                        env_selection=env_selection,
                        fixtures_dir=fixtures_dir,
                        diagnostics=diagnostics,
                        attack_env_seed=attack_seed,
                        attack_run_config=AttackRunConfig(
                            time_budget_s=budget_s,
                            max_tool_hops=max_tool_hops,
                        ),
                    )
            finally:
                unload_gguf_server(server, label=model)
        except TimeoutError as err:
            summary = {
                "model": model,
                "score_normalized_0_to_1000": 0.0,
                "score_raw": 0.0,
                "findings_count": 0,
                "unique_cells": 0,
                "unique_canonical_cells": 0,
                "evaluation_time_s": time.time() - started_at,
                "wall_time_s": time.time() - started_at,
                "guardrail_id": None,
                "guardrail_version": None,
                "env": env_selection.value,
                "seed": attack_seed,
                "budget_s": budget_s,
                "backend": "gguf_llama_cpp",
                "error": str(err),
                "error_type": type(err).__name__,
            }
            (model_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(
                f"[{model}] evaluation TIMED OUT ({err}) -- generation likely handed "
                f"replay more candidates than its own {budget_s:.0f}s budget could "
                f"get through. Recorded as a zero-score row; see {model_dir}/summary.json."
            )
            summaries.append(summary)
            continue

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
            "seed": attack_seed,
            "budget_s": budget_s,
            "backend": "gguf_llama_cpp",
        }
        (model_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        _write_findings(model_dir / "findings.jsonl", attack.findings)
        summaries.append(summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))

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
