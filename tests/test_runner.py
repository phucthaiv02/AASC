from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from aicomp_sdk.core.env.api import EnvSelection

from aas_nim_validation.config import Settings
from aas_nim_validation import runner


class FinalResponse:
    choices = [
        SimpleNamespace(
            message=SimpleNamespace(content="I cannot help with that.", tool_calls=None)
        )
    ]

    def model_dump(self, **_kwargs):
        return {"choices": [{"message": {"content": "I cannot help with that."}}]}


class FakeOpenAI:
    def __init__(self, **_kwargs):
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_request: FinalResponse())
        )


def test_runner_smoke_without_network(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "OpenAI", FakeOpenAI)
    settings = Settings(
        base_url="http://localhost:8000/v1",
        api_key="not-used",
        models=("test/model",),
        budget_s=1.0,
        max_tool_hops=2,
        attack_seed=123,
        max_tokens=32,
        temperature=0.0,
        timeout_s=5.0,
        max_retries=0,
    )

    result = runner.validate_models(
        attack_path=Path(__file__).parents[1] / "solution" / "template.py",
        artifacts_dir=tmp_path,
        settings=settings,
        env_selection=EnvSelection.GYM,
    )

    assert result["local_public_mean"] == 0.0
    assert (tmp_path / "test_model" / "summary.json").is_file()
    assert (tmp_path / "summary.json").is_file()


def test_validate_models_survives_a_timeout_from_evaluate_redteam(monkeypatch, tmp_path):
    """evaluate_redteam raises a bare TimeoutError when either the generation
    or replay phase overruns its own budget window (see runner.py's
    _validate_one_model docstring/comment). Before this fix that exception
    was uncaught and could crash the whole validate_models() call -- this
    confirms it's now caught, recorded as a zero-score/errored row, and
    doesn't stop other models or corrupt the combined summary."""
    monkeypatch.setattr(runner, "OpenAI", FakeOpenAI)

    def _raise_timeout(*_args, **_kwargs):
        raise TimeoutError("attack replay exceeded its time budget")

    monkeypatch.setattr(runner, "evaluate_redteam", _raise_timeout)

    settings = Settings(
        base_url="http://localhost:8000/v1",
        api_key="not-used",
        models=("test/model",),
        budget_s=1.0,
        max_tool_hops=2,
        attack_seed=123,
        max_tokens=32,
        temperature=0.0,
        timeout_s=5.0,
        max_retries=0,
    )

    result = runner.validate_models(
        attack_path=Path(__file__).parents[1] / "solution" / "template.py",
        artifacts_dir=tmp_path,
        settings=settings,
        env_selection=EnvSelection.GYM,
    )

    assert result["local_public_mean"] == 0.0
    summary_path = tmp_path / "test_model" / "summary.json"
    assert summary_path.is_file()
    summary = json.loads(summary_path.read_text())
    assert summary["score_normalized_0_to_1000"] == 0.0
    assert summary["error_type"] == "TimeoutError"
    assert "time budget" in summary["error"]
