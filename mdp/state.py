from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


KILL_CHAIN_PHASES = [
    "reconnaissance",
    "weaponization",
    "delivery",
    "exploitation",
    "installation",
    "command_control",
    "actions_objectives",
]

NUM_PHASES = len(KILL_CHAIN_PHASES)


@dataclass
class AttackState:
    host_compromised: np.ndarray
    vuln_cvss: np.ndarray
    phase_complete: np.ndarray
    defense_posture: np.ndarray
    host_values: np.ndarray
    step: int = 0

    def __post_init__(self) -> None:
        assert self.phase_complete.shape == (NUM_PHASES,)
        assert self.host_compromised.shape == self.host_values.shape

    def is_terminal(self) -> bool:
        return bool(np.all(self.phase_complete))

    def to_vector(self) -> np.ndarray:
        return np.concatenate([
            self.host_compromised.astype(np.float32),
            self.vuln_cvss.astype(np.float32),
            self.phase_complete.astype(np.float32),
            self.defense_posture.astype(np.float32),
        ])

    def copy(self) -> "AttackState":
        return AttackState(
            host_compromised=self.host_compromised.copy(),
            vuln_cvss=self.vuln_cvss.copy(),
            phase_complete=self.phase_complete.copy(),
            defense_posture=self.defense_posture.copy(),
            host_values=self.host_values.copy(),
            step=self.step,
        )


@dataclass
class AttackAction:
    agent_name: str
    technique_id: str
    target_host: str
    exploit_name: Optional[str] = None
    parameters: Dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"<{self.agent_name} | {self.technique_id} | {self.target_host}>"


@dataclass
class TransitionProbs:
    p_exploit: float
    p_evasion: float
    p_lateral: float

    def combined(self) -> float:
        return self.p_exploit * self.p_evasion * self.p_lateral


class RewardFunction:
    def __init__(self, alpha: float, beta: float, lam: float) -> None:
        self.alpha = alpha
        self.beta = beta
        self.lam = lam

    def __call__(
        self,
        prev_state: AttackState,
        next_state: AttackState,
    ) -> float:
        delta_phases = float(
            np.sum(next_state.phase_complete) - np.sum(prev_state.phase_complete)
        )
        host_value = float(
            np.dot(next_state.host_compromised, next_state.host_values)
        )
        defense_penalty = float(np.sum(next_state.defense_posture))
        return (
            self.alpha * delta_phases
            + self.beta * host_value
            - self.lam * defense_penalty
        )


class InformationGain:
    def compute(
        self,
        prior_belief: np.ndarray,
        posterior_belief: np.ndarray,
    ) -> float:
        eps = 1e-10
        p = np.clip(prior_belief, eps, 1 - eps)
        q = np.clip(posterior_belief, eps, 1 - eps)
        h_prior = -np.sum(p * np.log2(p) + (1 - p) * np.log2(1 - p))
        h_post = -np.sum(q * np.log2(q) + (1 - q) * np.log2(1 - q))
        return float(h_prior - h_post)


def compute_state_space_size(
    n_hosts: int,
    n_vulns: int,
    n_sensors: int,
) -> int:
    return (2 ** n_hosts) * (10 ** n_vulns) * (2 ** NUM_PHASES)


def build_initial_state(
    n_hosts: int,
    n_vulns: int,
    n_sensors: int,
    host_values: Optional[np.ndarray] = None,
) -> AttackState:
    if host_values is None:
        rng = np.random.default_rng(42)
        host_values = rng.uniform(0.1, 1.0, n_hosts)
    return AttackState(
        host_compromised=np.zeros(n_hosts, dtype=np.float32),
        vuln_cvss=np.zeros(n_vulns, dtype=np.float32),
        phase_complete=np.zeros(NUM_PHASES, dtype=np.float32),
        defense_posture=np.full(n_sensors, 0.1, dtype=np.float32),
        host_values=host_values.astype(np.float32),
    )
