import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from agent.agent import Agent
from agent.utils.scheduler import LinearScheduler


class AtariDQNAgent(Agent):
    def setup(self, config):
        # Ref: https://www.nature.com/articles/nature14236
        self.q_network = AtariValueNetwork(n_actions=self.action_space.n).to(self.device) # type: ignore
        self.q_target = AtariValueNetwork(n_actions=self.action_space.n).to(self.device) # type: ignore
        self.optim = Adam(self.q_network.parameters(), lr=0.00025, betas=(0.95, 0.95), eps=0.01)
        self.gamma = config["gamma"]

        self.scheduler = LinearScheduler(100000, 1, 0.1)
        self.num_actions = 0

        self.num_updates = 0
        self.target_update_freq = 10000

    def act(self, state, train):
        # Epsilon greedy
        if train: # and torch.rand(()) <= self.scheduler.value(self.num_actions):
            return self.action_space.sample()

        self.num_actions += 1

        # TODO: Implement frame stacking

        state = torch.tensor(np.array(state), dtype=torch.float32, device=self.device).unsqueeze(0)
        q_values = self.q_network(state)
        return np.int64(torch.argmax(q_values).cpu())

    def train(self, s_batch, a_batch, r_batch, s_next_batch, terminal_batch):
        # Q(s, a)
        q_values = self.q_network(s_batch)
        q_values = q_values[torch.arange(q_values.shape[0]).long(), a_batch.long()]

        # max Q(s', a')
        q_next_values = self.q_target(s_next_batch).max(1).values
        target = r_batch + self.gamma * q_next_values

        # Compute error
        error = torch.square(target - q_values) * (1 - terminal_batch.float())
        loss = torch.mean(error)

        # Update weights
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # Periodically update target network
        self.num_updates += 1
        if self.num_updates % self.target_update_freq == 0:
            self.q_target.load_state_dict(self.q_network.state_dict())

    def log(self, state, action):
        pass


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
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, state):
        state = state.float() / 255.0

        return self.net(state)
