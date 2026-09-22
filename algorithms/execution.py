from __future__ import annotations
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from mdp.state import AttackState, AttackAction, RewardFunction
from algorithms.ecs import ExploitChain, ExploitChainScore
from ppo.policy import PPOPolicy
from ppo.trainer import PPOTrainer


@dataclass
class ExecutionRecord:
    cycle: int
    action: AttackAction
    success: bool
    esi: float
    defense_shift: float
    replanned: bool
    reward: float


@dataclass
class KillChainResult:
    phase_vector: np.ndarray
    attack_success_rate: float
    phases_completed: int
    total_cycles: int
    replanning_events: int
    records: List[ExecutionRecord]
    time_to_objective: float


class DefenseDetector:
    def __init__(
        self,
        esi_threshold: float = 0.4,
        shift_threshold: float = 0.15,
        ema_alpha: float = 0.2,
    ) -> None:
        self.esi_threshold = esi_threshold
        self.shift_threshold = shift_threshold
        self.ema_alpha = ema_alpha
        self._ema_defense: Optional[np.ndarray] = None

    def update(self, defense_posture: np.ndarray) -> None:
        if self._ema_defense is None:
            self._ema_defense = defense_posture.copy()
        else:
            self._ema_defense = (
                self.ema_alpha * defense_posture
                + (1.0 - self.ema_alpha) * self._ema_defense
            )

    def triggered(self, esi: float, defense_posture: np.ndarray) -> bool:
        if esi < self.esi_threshold:
            return True
        if self._ema_defense is not None:
            shift = float(np.linalg.norm(defense_posture - self._ema_defense))
            if shift > self.shift_threshold:
                return True
        return False

    def defense_shift(self, defense_posture: np.ndarray) -> float:
        if self._ema_defense is None:
            return 0.0
        return float(np.linalg.norm(defense_posture - self._ema_defense))


class AdaptiveKillChainExecutor:
    def __init__(
        self,
        orchestrator,
        ppo_policy: PPOPolicy,
        ppo_trainer: PPOTrainer,
        reward_fn: RewardFunction,
        ecs: ExploitChainScore,
        max_cycles: int = 20,
        max_retries: int = 5,
        gamma: float = 0.99,
    ) -> None:
        self.orchestrator = orchestrator
        self.ppo = ppo_policy
        self.trainer = ppo_trainer
        self.reward_fn = reward_fn
        self.ecs = ecs
        self.max_cycles = max_cycles
        self.max_retries = max_retries
        self.gamma = gamma
        self.detector = DefenseDetector()

    def execute(
        self,
        initial_state: AttackState,
        exploit_chain: ExploitChain,
        objectives: List[str],
        target_host: str,
    ) -> KillChainResult:
        state = initial_state.copy()
        records: List[ExecutionRecord] = []
        retry_counter = 0
        replanning_events = 0
        t_start = time.time()

        state_vec = state.to_vector()
        self.detector.update(state.defense_posture)

        for cycle in range(self.max_cycles):
            state_tensor = self.ppo.state_to_tensor(state_vec)
            action, log_prob, value = self.ppo.act(state_tensor)

            ecs_score = exploit_chain.score if exploit_chain else 0.0
            attack_action = self.orchestrator.select_action(state, ecs_score)
            self.orchestrator.memory.update_state("primary_target", target_host)

            task = self._build_task(attack_action, state, objectives)
            success, log, result = self.orchestrator.run(state, task)

            esi = 1.0 if success else 0.0
            next_defense = self._observe_defense(state)
            defense_shift = self.detector.defense_shift(next_defense)

            prev_state = state.copy()
            state = self._apply_transition(state, attack_action, success)
            reward = self.reward_fn(prev_state, state)

            self.detector.update(state.defense_posture)

            replanned = False
            if self.detector.triggered(esi, state.defense_posture):
                if retry_counter < self.max_retries:
                    state_tensor_new = self.ppo.state_to_tensor(state.to_vector())
                    action, log_prob, value = self.ppo.act(state_tensor_new)
                    attack_action = self.orchestrator.select_action(state, ecs_score * 0.8)
                    retry_counter += 1
                    replanning_events += 1
                    replanned = True
                else:
                    break

            self.trainer.store(
                state=state_vec,
                action=action,
                reward=reward,
                value=value,
                log_prob=log_prob,
                done=state.is_terminal(),
            )

            if self.trainer.should_update():
                self.trainer.update(self.ppo)

            self.orchestrator.record_episode(
                state_desc=f"cycle={cycle} phase={int(np.argmin(state.phase_complete))}",
                action=str(attack_action),
                outcome="success" if success else "failure",
                phase=int(np.sum(state.phase_complete)),
                success=success,
            )

            records.append(ExecutionRecord(
                cycle=cycle,
                action=attack_action,
                success=success,
                esi=esi,
                defense_shift=defense_shift,
                replanned=replanned,
                reward=reward,
            ))

            state_vec = state.to_vector()

            if state.is_terminal():
                break

        elapsed = time.time() - t_start
        phases_done = int(np.sum(state.phase_complete))
        total_phases = len(state.phase_complete)
        asr = phases_done / total_phases

        return KillChainResult(
            phase_vector=state.phase_complete.copy(),
            attack_success_rate=asr,
            phases_completed=phases_done,
            total_cycles=len(records),
            replanning_events=replanning_events,
            records=records,
            time_to_objective=elapsed / 3600.0,
        )

    def _build_task(
        self,
        action: AttackAction,
        state: AttackState,
        objectives: List[str],
    ) -> str:
        next_phase = int(np.argmin(state.phase_complete))
        obj_str = "; ".join(objectives[:3])
        return (
            f"Execute {action.agent_name} for phase index {next_phase}. "
            f"Target: {action.target_host}. "
            f"Objectives: {obj_str}. "
            f"Defense posture mean: {float(np.mean(state.defense_posture)):.2f}."
        )

    def _observe_defense(self, state: AttackState) -> np.ndarray:
        noise = np.random.default_rng().normal(0, 0.02, state.defense_posture.shape)
        return np.clip(state.defense_posture + noise, 0.0, 1.0)

    def _apply_transition(
        self,
        state: AttackState,
        action: AttackAction,
        success: bool,
    ) -> AttackState:
        new_state = state.copy()
        new_state.step += 1
        if success:
            next_phase = int(np.argmin(new_state.phase_complete))
            if next_phase < len(new_state.phase_complete):
                new_state.phase_complete[next_phase] = 1.0
            first_uncommitted = int(np.argmin(new_state.host_compromised))
            if first_uncommitted < len(new_state.host_compromised):
                new_state.host_compromised[first_uncommitted] = 1.0
        else:
            delta = np.random.default_rng().uniform(0.03, 0.08, new_state.defense_posture.shape)
            new_state.defense_posture = np.clip(new_state.defense_posture + delta, 0.0, 1.0)
        return new_state
