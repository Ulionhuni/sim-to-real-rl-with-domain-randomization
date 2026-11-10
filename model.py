"""
Sim-to-Real RL with Domain Randomization

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 7 - build_actor_network
import torch
import torch.nn as nn

def build_actor_network(obs_dim, action_dim, hidden_dim=64):
    """Build a Gaussian actor: forward(obs) -> (mean, std).

    Store the learnable log-std as an attribute named `log_std`
    (an nn.Parameter of shape (action_dim,), initialized to zeros) --
    later steps read it via `actor.log_std`.
    """
    # TODO: Construct a Gaussian actor network mapping obs to (mean, std)
    # with two Tanh hidden layers and std = exp(log_std).
    class ActorNetwork(nn.Module):
        def __init__(self):
            super().__init__()

            self.hidden_layers = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh()
            )
            self.mean = nn.Linear(hidden_dim, action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim))

        def forward(self, obs: torch.Tensor):
            mean = self.mean(self.hidden_layers(obs))
            std = torch.exp(self.log_std)

            return mean, std

    return ActorNetwork()

