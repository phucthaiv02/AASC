from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from aicomp_sdk.attacks.contracts import AttackRunConfig


def _load_adaptive_module():
    path = Path(__file__).parents[1] / "solution" / "adaptive.py"
    spec = importlib.util.spec_from_file_location("adaptive", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


mod = _load_adaptive_module()


def test_slack_tracker_extends_when_slack_is_abundant() -> None:
    tracker = mod._SlackTracker(assumed_extension_factor=2.0, safety_frac=0.85)

    assert tracker.should_extend(
        remaining_slots=10,
        wall_time_left=1000.0,
        replay_time_left=1000.0,
        replay_cost_coef=1.0,
        fallback_plain=1.0,
    )


def test_slack_tracker_refuses_when_time_bound() -> None:
    tracker = mod._SlackTracker(assumed_extension_factor=2.0, safety_frac=0.85)

    assert not tracker.should_extend(
        remaining_slots=1000,
        wall_time_left=10.0,
        replay_time_left=10.0,
        replay_cost_coef=1.0,
        fallback_plain=1.0,
    )


def test_slack_tracker_respects_replay_budget_independently_of_wall_budget() -> None:
    """Plenty of wall-clock slack must not matter if the REPLAY budget is tight --
    an overrun there voids the whole submission, the worse failure mode."""
    tracker = mod._SlackTracker(assumed_extension_factor=2.0, safety_frac=0.85)

    assert not tracker.should_extend(
        remaining_slots=10,
        wall_time_left=1000.0,
        replay_time_left=1.0,
        replay_cost_coef=1.0,
        fallback_plain=1.0,
    )


class _AlwaysFiresFakeEnv:
    """Near-instant env: every interaction is a successful marker post, so the
    only thing gating candidate count/extension is the algorithm's own pacing
    logic, not real latency."""

    def __init__(self) -> None:
        self.seed = 0

    def reset(self) -> None:
        pass

    def interact(self, message: str, max_tool_hops: int = 8) -> None:
        del message, max_tool_hops

    def export_trace_dict(self) -> dict[str, Any]:
        return {
            "tool_events": [
                {"ok": True, "name": "http.post", "args": {"data": mod.PAYLOAD}}
            ]
        }


def test_adaptive_extends_once_cap_bound_slack_is_established() -> None:
    """With a near-instant env, a modest cap, and a generous budget, the run is
    clearly cap-bound with abundant slack -- most candidates past the bandit's
    forced-exploration window should pick up the free second turn."""
    algorithm = mod.AttackAlgorithm(
        {"hard_n_cap": 30, "slowest0": 0.0, "bandit_min_trials": 2}
    )

    candidates = algorithm.run(_AlwaysFiresFakeEnv(), AttackRunConfig(time_budget_s=50.0))

    turn_counts = [len(c.user_messages) for c in candidates]
    assert len(candidates) == 30
    assert 1 in turn_counts  # forced-exploration candidates stay single-turn
    assert 2 in turn_counts  # slack was detected and banked as free severity


def test_adaptive_confused_deputy_false_never_extends() -> None:
    algorithm = mod.AttackAlgorithm(
        {
            "hard_n_cap": 20,
            "slowest0": 0.0,
            "bandit_min_trials": 2,
            "adaptive_confused_deputy": False,
        }
    )

    candidates = algorithm.run(_AlwaysFiresFakeEnv(), AttackRunConfig(time_budget_s=50.0))

    assert candidates
    assert all(len(c.user_messages) == 1 for c in candidates)
