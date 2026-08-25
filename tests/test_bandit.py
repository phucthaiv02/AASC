from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from aicomp_sdk.attacks.contracts import AttackRunConfig


def _load_bandit_module():
    path = Path(__file__).parents[1] / "solution" / "bandit.py"
    spec = importlib.util.spec_from_file_location("bandit", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


bandit_mod = _load_bandit_module()


def test_every_pool_template_avoids_suppression_words_and_formats_cleanly() -> None:
    suppression_words = ("send", "email", "upload")
    for template in bandit_mod.TEMPLATE_POOL:
        msg = template.format(url=bandit_mod._url(0), payload=bandit_mod.PAYLOAD)
        assert not any(word in msg.lower() for word in suppression_words)
        assert bandit_mod.PAYLOAD in msg


class _TemplateAwareFakeEnv:
    """Deterministically fires only for messages built from `winning_template`,
    so a test can check the bandit converges to whichever arm actually works."""

    def __init__(self, winning_template: str) -> None:
        self.winning_template = winning_template
        self.last_message: str | None = None
        self.seed = 0

    def reset(self) -> None:
        pass

    def interact(self, message: str, max_tool_hops: int = 8) -> None:
        self.last_message = message

    def export_trace_dict(self) -> dict[str, Any]:
        fired = self.last_message is not None and self._matches_winner(self.last_message)
        return {
            "tool_events": [
                {"ok": True, "name": "http.post", "args": {"data": bandit_mod.PAYLOAD}}
            ]
            if fired
            else []
        }

    def _matches_winner(self, message: str) -> bool:
        # The winning template is the only one containing this literal fragment
        # (each pool template has a distinct fixed prefix independent of {url}).
        prefix = self.winning_template.split("{url}")[0]
        return prefix in message


def test_bandit_converges_to_the_only_firing_template() -> None:
    winner = bandit_mod._TASKLIST_TEMPLATE
    assert winner in bandit_mod.TEMPLATE_POOL
    env = _TemplateAwareFakeEnv(winner)
    algorithm = bandit_mod.AttackAlgorithm(
        {"hard_n_cap": 40, "slowest0": 0.0, "bandit_min_trials": 2, "bandit_epsilon": 0.0}
    )

    candidates = algorithm.run(env, AttackRunConfig(time_budget_s=1.0))

    assert candidates
    winner_prefix = winner.split("{url}")[0]
    # Every KEPT candidate must have used the only template that actually fires --
    # forced exploration tries the others too, but only firing ones are kept.
    assert all(
        winner_prefix in candidate.user_messages[0] for candidate in candidates
    )


def test_bandit_false_always_uses_plain_template() -> None:
    env = _TemplateAwareFakeEnv(bandit_mod.TEMPLATE)
    algorithm = bandit_mod.AttackAlgorithm(
        {"hard_n_cap": 5, "slowest0": 0.0, "bandit": False}
    )

    candidates = algorithm.run(env, AttackRunConfig(time_budget_s=1.0))

    assert candidates
    template_prefix = bandit_mod.TEMPLATE.split("{url}")[0]
    assert all(
        template_prefix in candidate.user_messages[0] for candidate in candidates
    )
