from __future__ import annotations
import torch
import numpy as np
from typing import Tuple
from ppo.network import ActorCriticNetwork
from config import PPOConfig


class PPOPolicy:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        ppo_config: PPOConfig,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.network = ActorCriticNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=ppo_config.hidden_dim,
            num_layers=ppo_config.num_layers,
        ).to(self.device)
        self.action_dim = action_dim

    def state_to_tensor(self, state_vec: np.ndarray) -> torch.Tensor:
        return torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)

    def act(
        self,
        state_tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            action, log_prob, value = self.network.act(state_tensor)
        return action, log_prob, value

    def evaluate(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.network.evaluate(states, actions)

    def save(self, path: str) -> None:
        torch.save(self.network.state_dict(), path)

    def load(self, path: str) -> None:
        self.network.load_state_dict(torch.load(path, map_location=self.device))

    def parameters(self):
        return self.network.parameters()
