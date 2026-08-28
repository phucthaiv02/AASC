from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from aicomp_sdk.attacks.contracts import AttackRunConfig


def _load_blast_module():
    path = Path(__file__).parents[1] / "solution" / "blast.py"
    spec = importlib.util.spec_from_file_location("blast", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


blast_mod = _load_blast_module()


def test_template_avoids_suppression_words_and_formats_cleanly() -> None:
    suppression_words = ("send", "email", "upload")
    msg = blast_mod.TEMPLATE.format(url=blast_mod._url(0), payload=blast_mod.PAYLOAD)
    assert not any(word in msg.lower() for word in suppression_words)
    assert blast_mod.PAYLOAD in msg


class _CountingAlwaysFiresEnv:
    """Every interact() fires. Counts total interact() calls so a test can
    assert the generation-phase real interact() count stays bounded by the
    probe count, regardless of how many candidates are ultimately emitted."""

    def __init__(self) -> None:
        self.interact_count = 0
        self.seed = 0

    def reset(self) -> None:
        pass

    def interact(self, message: str, max_tool_hops: int = 8) -> None:
        self.interact_count += 1

    def export_trace_dict(self) -> dict[str, Any]:
        return {
            "tool_events": [
                {"ok": True, "name": "http.post", "args": {"data": blast_mod.PAYLOAD}}
            ]
        }


def test_generation_phase_interact_count_is_bounded_by_probe_n_not_by_cap() -> None:
    env = _CountingAlwaysFiresEnv()
    algorithm = blast_mod.AttackAlgorithm({"hard_n_cap": 2000, "probe_n": 15})

    candidates = algorithm.run(env, AttackRunConfig(time_budget_s=9000.0))

    assert len(candidates) == 2000
    # 1 untimed warmup + 15 probes == 16 total interact() calls, no matter how
    # many candidates get blind-emitted afterwards.
    assert env.interact_count == 1 + 15


def test_all_emitted_candidates_have_distinct_messages() -> None:
    env = _CountingAlwaysFiresEnv()
    algorithm = blast_mod.AttackAlgorithm({"hard_n_cap": 50, "probe_n": 5})

    candidates = algorithm.run(env, AttackRunConfig(time_budget_s=9000.0))

    messages = [c.user_messages[0] for c in candidates]
    assert len(messages) == len(set(messages)) == 50


class _NeverFiresEnv:
    def __init__(self) -> None:
        self.seed = 0

    def reset(self) -> None:
        pass

    def interact(self, message: str, max_tool_hops: int = 8) -> None:
        pass

    def export_trace_dict(self) -> dict[str, Any]:
        return {"tool_events": []}


def test_never_fires_falls_back_to_fallback_n() -> None:
    env = _NeverFiresEnv()
    algorithm = blast_mod.AttackAlgorithm({"probe_n": 5})

    candidates = algorithm.run(env, AttackRunConfig(time_budget_s=9000.0))

    assert len(candidates) == blast_mod.FALLBACK_N


class _PartialFireEnv:
    """First call (warmup) never fires. Of the probe calls after it, only
    even-numbered ones fire -- exactly half, so a test can check a fire rate
    under 100% (but above the threshold) still triggers a full blind blast."""

    def __init__(self) -> None:
        self.calls = 0
        self.seed = 0

    def reset(self) -> None:
        pass

    def interact(self, message: str, max_tool_hops: int = 8) -> None:
        self.calls += 1

    def export_trace_dict(self) -> dict[str, Any]:
        fired = self.calls > 1 and self.calls % 2 == 0
        return {
            "tool_events": [
                {"ok": True, "name": "http.post", "args": {"data": blast_mod.PAYLOAD}}
            ]
            if fired
            else []
        }


def test_partial_fire_rate_above_threshold_still_blasts_full_cap() -> None:
    env = _PartialFireEnv()
    # 10 probes, 5 of them fire (every other one) -- above the 3-fire threshold.
    algorithm = blast_mod.AttackAlgorithm(
        {"hard_n_cap": 100, "probe_n": 10, "probe_min_fires": 3}
    )

    candidates = algorithm.run(env, AttackRunConfig(time_budget_s=9000.0))

    # Blind emission doesn't re-check firing -- it blasts the full cap once
    # the probe confirms the template is viable; replay decides the rest.
    assert len(candidates) == 100


def test_fire_rate_below_threshold_falls_back_to_fallback_n() -> None:
    env = _PartialFireEnv()
    # Same env (5/10 probes fire) but the threshold now demands more fires
    # than the template actually delivers.
    algorithm = blast_mod.AttackAlgorithm(
        {"hard_n_cap": 100, "probe_n": 10, "probe_min_fires": 6}
    )

    candidates = algorithm.run(env, AttackRunConfig(time_budget_s=9000.0))

    assert len(candidates) == blast_mod.FALLBACK_N


def test_env_none_falls_back_without_touching_env() -> None:
    algorithm = blast_mod.AttackAlgorithm({})
    candidates = algorithm.run(None, AttackRunConfig(time_budget_s=9000.0))
    assert len(candidates) == blast_mod.FALLBACK_N


def test_n_candidates_override_short_circuits_everything() -> None:
    algorithm = blast_mod.AttackAlgorithm({"n_candidates": 7})
    candidates = algorithm.run(None, AttackRunConfig(time_budget_s=9000.0))
    assert len(candidates) == 7
