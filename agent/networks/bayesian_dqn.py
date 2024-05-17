import torch
import torch.nn as nn
import torch.nn.functional as F
from agent.networks.bayesian import BayesianConv2d, BayesianLinear


class BayesianAtariDQNFeatures(nn.Module):
    """Convolutional part of the Atari DQN network to extract features from the state. This can be
    re-used in the networks of other models. This is used for the Bayesian version.

    Ref: https://www.nature.com/articles/nature14236

    Input: B x 4 x 84 x 84
    Output: B x 3136
    
    """

    def __init__(self):
        super(BayesianAtariDQNFeatures, self).__init__()

        self.conv1 = BayesianConv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = BayesianConv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = BayesianConv2d(64, 64, kernel_size=3, stride=1)

    def forward(self, state: torch.Tensor, train=True) -> torch.Tensor:
        # Ensure the input is in [0,1]
        if state.dtype == torch.uint8:
            state = state.float() / 255.0

        x = self.conv1(state, train)
        x = F.relu(x)
        x = self.conv2(x, train)
        x = F.relu(x)
        x = self.conv3(x, train)
        x = F.relu(x)

        return x.flatten(start_dim=1)

    def prior_kl_loss(self):
        kl1 = self.conv1.prior_kl_loss()
        kl2 = self.conv2.prior_kl_loss()
        kl3 = self.conv3.prior_kl_loss()
        return kl1 + kl2 + kl3


class BayesianQNetwork(nn.Module):
    """Linear network that maps convolutional features of the Atari DQN network to Q-values per
    state. This is used for the Bayesian version.

    Ref: https://www.nature.com/articles/nature14236

    Input: B x D
    Output: B x A

    """

    def __init__(self, input_dim: int, hidden_dim: int, n_actions: int):
        super(BayesianQNetwork, self).__init__()

        self.linear1 = BayesianLinear(input_dim, hidden_dim)
        self.linear2 = BayesianLinear(hidden_dim, n_actions)

    def forward(self, state: torch.Tensor, train=True) -> torch.Tensor:
        x = self.linear1(state, train)
        x = F.relu(x)
        x = self.linear2(x, train)

        return x

    def prior_kl_loss(self):
        kl1 = self.linear1.prior_kl_loss()
        kl2 = self.linear2.prior_kl_loss()
        return kl1 + kl2
