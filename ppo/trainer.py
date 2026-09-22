from __future__ import annotations
import torch
import torch.optim as optim
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from ppo.policy import PPOPolicy
from config import PPOConfig


@dataclass
class Trajectory:
    states: List[np.ndarray] = field(default_factory=list)
    actions: List[torch.Tensor] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    values: List[torch.Tensor] = field(default_factory=list)
    log_probs: List[torch.Tensor] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.rewards)

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()


class PPOTrainer:
    def __init__(
        self,
        policy: PPOPolicy,
        config: PPOConfig,
        device: str = "cpu",
    ) -> None:
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.optimizer = optim.Adam(policy.parameters(), lr=config.learning_rate)
        self.trajectory = Trajectory()
        self._update_interval = config.batch_size

    def store(
        self,
        state: np.ndarray,
        action: torch.Tensor,
        reward: float,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        done: bool,
    ) -> None:
        self.trajectory.states.append(state)
        self.trajectory.actions.append(action)
        self.trajectory.rewards.append(reward)
        self.trajectory.values.append(value)
        self.trajectory.log_probs.append(log_prob)
        self.trajectory.dones.append(done)

    def should_update(self) -> bool:
        return len(self.trajectory) >= self._update_interval

    def _compute_returns_and_advantages(
        self,
        rewards: List[float],
        values: List[torch.Tensor],
        dones: List[bool],
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n = len(rewards)
        returns = torch.zeros(n, device=self.device)
        advantages = torch.zeros(n, device=self.device)
        gae = 0.0
        next_value = 0.0

        for t in reversed(range(n)):
            mask = 0.0 if dones[t] else 1.0
            delta = rewards[t] + gamma * next_value * mask - values[t].item()
            gae = delta + gamma * lam * mask * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t].item()
            next_value = values[t].item()

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def update(self, policy: PPOPolicy) -> Dict:
        traj = self.trajectory
        gamma = self.config.gamma
        clip_eps = self.config.clip_epsilon

        states_t = torch.FloatTensor(np.stack(traj.states)).to(self.device)
        actions_t = torch.stack(traj.actions).to(self.device).squeeze(-1)
        old_log_probs_t = torch.stack(traj.log_probs).to(self.device).squeeze(-1)

        returns_t, advantages_t = self._compute_returns_and_advantages(
            traj.rewards, traj.values, traj.dones, gamma=gamma
        )

        policy_losses, value_losses, entropy_losses = [], [], []

        for _ in range(self.config.epochs_per_update):
            idx = np.random.permutation(len(traj))
            for start in range(0, len(traj), self.config.batch_size):
                batch = idx[start: start + self.config.batch_size]
                if len(batch) == 0:
                    continue

                b_states = states_t[batch]
                b_actions = actions_t[batch]
                b_old_lp = old_log_probs_t[batch]
                b_returns = returns_t[batch]
                b_adv = advantages_t[batch]

                new_log_probs, values, entropy = policy.evaluate(b_states, b_actions)
                ratio = torch.exp(new_log_probs - b_old_lp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = ((values - b_returns) ** 2).mean()
                entropy_loss = -entropy.mean()
                loss = (
                    policy_loss
                    + self.config.value_coeff * value_loss
                    + self.config.entropy_coeff * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())

        self.trajectory.clear()
        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy_loss": float(np.mean(entropy_losses)),
        }


from typing import Dict
