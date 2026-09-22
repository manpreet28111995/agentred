from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class ActorCriticNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        layers = [nn.Linear(state_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        self.shared = nn.Sequential(*layers)
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(x)
        logits = self.actor_head(features)
        value = self.critic_head(features)
        return logits, value

    def act(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value.squeeze(-1)

    def evaluate(
        self,
        x: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, value.squeeze(-1), entropy
