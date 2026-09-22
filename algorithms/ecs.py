from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from mdp.state import AttackState


@dataclass
class ExploitEntry:
    cve_id: str
    msf_module: str
    cvss_base: float
    cvss_exploitability: float
    cwe_id: str
    attack_technique: str
    target_host: str
    target_port: int


@dataclass
class ExploitChain:
    entries: List[ExploitEntry]
    score: float
    detection_probs: List[float]

    def __len__(self) -> int:
        return len(self.entries)

    def success_probability(self) -> float:
        probs = [1.0 - d for d in self.detection_probs]
        result = 1.0
        for p in probs:
            result *= p
        return result


class ExploitChainScore:
    def __init__(self, rho: float = 0.35) -> None:
        if not 0.0 < rho < 1.0:
            raise ValueError("rho must be in (0, 1)")
        self.rho = rho

    def _position_weights(self, n: int) -> np.ndarray:
        unnormalised = np.array([(1.0 - self.rho) ** i for i in range(n)], dtype=np.float64)
        return unnormalised / unnormalised.sum()

    def compute(
        self,
        chain: List[ExploitEntry],
        detection_probs: List[float],
    ) -> float:
        if not chain:
            return 0.0
        n = len(chain)
        weights = self._position_weights(n)
        score = 0.0
        for i, (entry, delta, w) in enumerate(zip(chain, detection_probs, weights)):
            score += w * entry.cvss_exploitability * (1.0 - delta) * entry.cvss_base
        return float(score)

    def optimal_chain_length_bound(
        self,
        cvss_min: float,
        phi_min: float,
    ) -> int:
        if cvss_min <= 0.0 or phi_min <= 0.0:
            return 1
        numerator = math.log(cvss_min * phi_min + 1e-10)
        denominator = math.log(1.0 - self.rho + 1e-10)
        if denominator >= 0:
            return 1
        return max(1, math.ceil(numerator / denominator))

    def rank_chains(
        self,
        candidates: List[List[ExploitEntry]],
        detection_probs_per_chain: List[List[float]],
    ) -> List[Tuple[ExploitChain, float]]:
        scored = []
        for chain, dets in zip(candidates, detection_probs_per_chain):
            s = self.compute(chain, dets)
            scored.append(ExploitChain(entries=chain, score=s, detection_probs=dets))
        scored.sort(key=lambda x: x.score, reverse=True)
        return [(c, c.score) for c in scored]

    def estimate_detection_probability(
        self,
        exploit: ExploitEntry,
        defense_posture: np.ndarray,
    ) -> float:
        base = 0.1 + (0.9 - exploit.cvss_exploitability / 10.0) * 0.4
        defense_factor = float(np.mean(defense_posture))
        return min(base * (1.0 + defense_factor), 0.95)


class ExploitChainEnumerator:
    def __init__(
        self,
        ecs: ExploitChainScore,
        cvss_threshold: float = 7.0,
    ) -> None:
        self.ecs = ecs
        self.cvss_threshold = cvss_threshold

    def enumerate_chains(
        self,
        exploit_candidates: List[ExploitEntry],
        state: AttackState,
    ) -> ExploitChain:
        filtered = [
            e for e in exploit_candidates
            if e.cvss_base >= self.cvss_threshold
        ]
        if not filtered:
            filtered = exploit_candidates

        cvss_values = [e.cvss_base for e in filtered]
        phi_values = [e.cvss_exploitability for e in filtered]
        cvss_min = min(cvss_values) if cvss_values else 1.0
        phi_min = min(phi_values) if phi_values else 0.1

        n_star = self.ecs.optimal_chain_length_bound(cvss_min, phi_min)
        n_star = max(1, min(n_star, len(filtered)))

        best_chain: Optional[ExploitChain] = None
        best_score = -1.0

        for length in range(1, n_star + 1):
            for combo in self._combinations(filtered, length):
                dets = [
                    self.ecs.estimate_detection_probability(e, state.defense_posture)
                    for e in combo
                ]
                score = self.ecs.compute(combo, dets)
                if score > best_score:
                    best_score = score
                    best_chain = ExploitChain(
                        entries=list(combo),
                        score=score,
                        detection_probs=dets,
                    )

        if best_chain is None:
            single = filtered[0] if filtered else exploit_candidates[0]
            dets = [self.ecs.estimate_detection_probability(single, state.defense_posture)]
            score = self.ecs.compute([single], dets)
            best_chain = ExploitChain(entries=[single], score=score, detection_probs=dets)

        return best_chain

    @staticmethod
    def _combinations(items: List, r: int):
        if r == 0:
            yield []
            return
        if r > len(items):
            return
        for i, item in enumerate(items):
            for rest in ExploitChainEnumerator._combinations(items[i + 1:], r - 1):
                yield [item] + rest
