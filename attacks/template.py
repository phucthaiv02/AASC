"""Minimal template for a new AAS attack variant."""

from __future__ import annotations

from aicomp_sdk.attacks import AttackAlgorithmBase, AttackCandidate, AttackRunConfig


class AttackAlgorithm(AttackAlgorithmBase):
    def run(self, env, config: AttackRunConfig) -> list[AttackCandidate]:
        del env, config
        return []

