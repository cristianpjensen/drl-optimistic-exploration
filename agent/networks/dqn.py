import torch
import torch.nn as nn
from agent.networks.weight_init import init_weights


class AtariDQNFeatures(nn.Module):
    """Convolutional part of the Atari DQN network to extract features from the state. This can be
    re-used in the networks of other models.

    Ref: https://www.nature.com/articles/nature14236

    Input: B x 4 x 84 x 84
    Output: B x 3136
    
    """

    def __init__(self):
        super(AtariDQNFeatures, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),  # 32 x 20 x 20
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # 64 x 9 x 9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # 64 x 7 x 7
            nn.ReLU(),
            nn.Flatten(),
        )
        self.net.apply(init_weights)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Ensure the input is in [0,1]
        if state.dtype == torch.uint8:
            state = state.float() / 255.0

        return self.net(state)


class QNetwork(nn.Module):
    """Linear network that maps convolutional features of the Atari DQN network to Q-values per
    state.

    Ref: https://www.nature.com/articles/nature14236

    Input: B x D
    Output: B x A

    """

    def __init__(self, input_dim: int, hidden_dim: int, n_actions: int):
        super(QNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )
        self.net.apply(init_weights)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)
