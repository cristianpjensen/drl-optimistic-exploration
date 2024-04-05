import torch
from torch.optim import Adam
import numpy as np

from agent.agent import Agent
from agent.models.value_network import AtariValueNetwork


class AtariDQNAgent(Agent):
    def setup(self, config, device):
        # Ref: https://www.nature.com/articles/nature14236
        self.q_network = AtariValueNetwork(n_actions=self.action_space.n).to(device) # type: ignore
        self.optim = Adam(self.q_network.parameters(), lr=0.00025, betas=(0.95, 0.95), eps=0.01)
        self.gamma = config["gamma"]

    def act(self, state, timestep, train):
        # Epsilon greedy
        if train and torch.rand(()) <= 0.1:
            return np.int64(self.action_space.sample())

        state = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0)
        q_values = self.q_network(state)
        return np.int64(torch.argmax(q_values))

    def train(self, s_batch, a_batch, r_batch, s_next_batch):
        # Q(s, a)
        q_values = self.q_network(s_batch)
        q_values = q_values[torch.arange(q_values.shape[0]), a_batch.int()]

        # max Q(s', a')
        q_next_values = self.q_network(s_next_batch)
        q_next_values = torch.max(q_next_values, dim=1).values

        target = r_batch + self.gamma * q_next_values
        loss = torch.mean((q_values - target) ** 2)

        # Update weights
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def log(self, state, action):
        pass
