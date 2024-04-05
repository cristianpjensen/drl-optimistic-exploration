import torch.nn as nn


class AtariValueNetwork(nn.Module):
    """ Ref: https://www.nature.com/articles/nature14236 """

    def __init__(self, n_actions: int):
        super(AtariValueNetwork, self).__init__()

        # Input: 4 x 84 x 84

        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),  # Output: 32 x 20 x 20
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # Output: 64 x 9 x 9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # Output: 64 x 7 x 7
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        return self.net(x)
