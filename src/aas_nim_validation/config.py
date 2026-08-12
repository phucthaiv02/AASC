from __future__ import annotations

import os
import sys
import importlib.util
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODELS = ("openai/gpt-oss-20b", "google/gemma-4-31b-it")


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None else int(raw)


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None else float(raw)


def bootstrap_sdk_path() -> None:
    """Make the downloaded Kaggle competition SDK importable."""
    if importlib.util.find_spec("aicomp_sdk") is not None:
        return
    raw_path = os.getenv("AAS_SDK_PATH", "").strip()
    candidates = []
    if raw_path:
        candidates.append(Path(raw_path).expanduser())
    candidates.extend((Path.cwd() / "input", Path(__file__).resolve().parents[2] / "input"))

    for candidate in candidates:
        sdk_path = candidate.resolve()
        if (sdk_path / "aicomp_sdk").is_dir():
            value = str(sdk_path)
            if value not in sys.path:
                sys.path.insert(0, value)
            return

    if raw_path:
        raise RuntimeError(f"AAS_SDK_PATH must contain aicomp_sdk/: {candidates[0].resolve()}")
    raise RuntimeError(
        "Cannot find aicomp_sdk. Put the competition SDK in ./input, set AAS_SDK_PATH, "
        "or install the 'sdk' extra."
    )


@dataclass(frozen=True)
class Settings:
    base_url: str
    api_key: str
    models: tuple[str, ...]
    budget_s: float
    max_tool_hops: int
    attack_seed: int
    max_tokens: int
    temperature: float
    timeout_s: float
    max_retries: int

    @classmethod
    def from_env(cls) -> "Settings":
        raw_models = os.getenv("NIM_MODELS", ",".join(DEFAULT_MODELS))
        models = tuple(model.strip() for model in raw_models.split(",") if model.strip())
        if not models:
            raise RuntimeError("NIM_MODELS must contain at least one model ID")

        base_url = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")
        api_key = os.getenv("NIM_API_KEY", "").strip()
        is_local = base_url.startswith(("http://localhost", "http://127.0.0.1", "http://0.0.0.0"))
        if not api_key and not is_local:
            raise RuntimeError("NIM_API_KEY is required for a hosted NIM endpoint")

        return cls(
            base_url=base_url,
            api_key=api_key or "not-used",
            models=models,
            budget_s=_float_env("AAS_BUDGET_S", 60.0),
            max_tool_hops=_int_env("AAS_MAX_TOOL_HOPS", 8),
            attack_seed=_int_env("AAS_ATTACK_SEED", 123),
            max_tokens=_int_env("NIM_MAX_TOKENS", 4096),
            temperature=_float_env("NIM_TEMPERATURE", 0.0),
            timeout_s=_float_env("NIM_TIMEOUT_S", 180.0),
            max_retries=_int_env("NIM_MAX_RETRIES", 3),
        )
