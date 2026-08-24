from __future__ import annotations

import json
from pathlib import Path

from aicomp_sdk.core.env.api import EnvSelection

from aas_nim_validation.gguf_runner import GgufPerfConfig, validate_gguf_models


class FakeLlama:
    """Stands in for ``llama_cpp.Llama`` so this test needs no GPU/model weights."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def create_chat_completion(self, **_kwargs):
        return {
            "choices": [
                {
                    "message": {"content": "I cannot help with that.", "tool_calls": None},
                    "finish_reason": "stop",
                }
            ]
        }

    def close(self):
        pass


def test_validate_gguf_models_smoke_without_gpu(tmp_path):
    fake_weights = tmp_path / "fake.gguf"
    fake_weights.write_bytes(b"")

    result = validate_gguf_models(
        attack_path=Path(__file__).parents[1] / "solution" / "template.py",
        artifacts_dir=tmp_path / "artifacts",
        models=("gpt_oss",),
        budget_s=1.0,
        max_tool_hops=2,
        env_selection=EnvSelection.GYM,
        perf=GgufPerfConfig(n_batch=64, n_ubatch=64),
        model_paths={"gpt_oss": str(fake_weights)},
        llama_cls=FakeLlama,
    )

    assert result["local_public_mean"] == 0.0
    summary_path = tmp_path / "artifacts" / "gpt_oss" / "summary.json"
    assert summary_path.is_file()
    summary = json.loads(summary_path.read_text())
    assert summary["backend"] == "gguf_llama_cpp"
    assert (tmp_path / "artifacts" / "summary.json").is_file()
