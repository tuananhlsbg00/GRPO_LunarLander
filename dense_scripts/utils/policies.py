# <dense_script/utils/policies.py>
import torch, torch.nn as nn
from torch.distributions.categorical import Categorical


class SimpleGRPOPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def dist(self, x: torch.Tensor) -> Categorical:
        return Categorical(logits=self.forward(x))


class SharedActorCriticPolicy(nn.Module):
    """
    Minimal actor-critic network matching the style of SimpleGRPOPolicy.
    - Shared body
    - Separate heads for policy logits and value
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh()
        )
        self.pi_head = nn.Linear(hidden, act_dim)
        self.v_head = nn.Linear(hidden, 1)

        # keep kwargs for cloning convenience
        self._init_kwargs = {"hidden": hidden}

    def forward(self, x: torch.Tensor):
        """Return both logits and value."""
        h = self.shared(x)
        return self.pi_head(h), self.v_head(h).squeeze(-1)

    def dist(self, x: torch.Tensor) -> Categorical:
        logits, _ = self.forward(x)
        return Categorical(logits=logits)

    def value(self, x: torch.Tensor) -> torch.Tensor:
        _, v = self.forward(x)
        return v

class SimpleActorCriticPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.pi = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, act_dim)
        )
        self.v = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1)
        )
    def dist(self, x): return Categorical(logits=self.pi(x))
    def value(self, x): return self.v(x).squeeze(-1)


