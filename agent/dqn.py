import os

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from agent.agent import Agent
from agent.utils.scheduler import LinearScheduler


class AtariDQNAgent(Agent):
    def setup(self, config):
        # Ref: https://www.nature.com/articles/nature14236
        self.q_network = AtariValueNetwork(n_actions=self.action_space.n).to(self.device)
        self.q_target = AtariValueNetwork(n_actions=self.action_space.n).to(self.device).requires_grad_(False)
        self.q_target.load_state_dict(self.q_network.state_dict())

        # Params from https://www.nature.com/articles/nature14236
        self.optim = Adam(self.q_network.parameters(), lr=0.00025, betas=(0.95, 0.95), eps=0.01)
        self.scheduler = LinearScheduler(1_000_000, 1, 0.1)
        self.gamma = config["gamma"]

        self.num_actions = 0
        self.num_updates = 0

        # Update target network every 10_000 steps at batch_size=32 (https://www.nature.com/articles/nature14236).
        self.target_update_freq = 32_000 // self.replay_buffer.batch_size

        # For logging the loss
        self.current_loss = 0
        self.logged_loss = True

    def act(self, state, train):
        with torch.no_grad():
            state = torch.tensor(state, device=self.device)
            q_values = self.q_network(state)

        action = np.zeros(state.shape[0], dtype=self.action_space.dtype)
        for i in range(state.shape[0]):
            self.num_actions += 1

            if train and np.random.random() < self.scheduler.value(self.num_actions):
                action[i] = self.action_space.sample()
            else:
                action[i] = torch.argmax(q_values[i]).cpu().numpy()

        return action

    def train(self, s_batch, a_batch, r_batch, s_next_batch, terminal_batch):
        # Q(s, a)
        q_values = self.q_network(s_batch)
        q_value = q_values[torch.arange(q_values.shape[0]).long(), a_batch.long()]

        # Compute target value
        q_next_value = self.q_target(s_next_batch).max(1).values
        target = r_batch + (self.gamma * q_next_value) * (1 - terminal_batch.float())

        # Compute error
        error = torch.square(target - q_value).clip(-1, 1)
        loss = torch.mean(error)

        # Update weights
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # Periodically update target network
        self.num_updates += 1
        if self.num_updates % self.target_update_freq == 0:
            self.q_target.load_state_dict(self.q_network.state_dict())

        self.current_loss = loss
        self.logged_loss = False

    def log(self, run):
        if not self.logged_loss:
            run["train/loss"].append(step=self.num_updates, value=self.current_loss)
            self.logged_loss = True

    def save(self, dir) -> bool:
        os.makedirs(dir, exist_ok=True)
        torch.save(self.q_network.state_dict(), f"{dir}/q_network.pt")
        return True

    def load(self, dir):
        self.q_target.load_state_dict(torch.load(f"{dir}/q_network.pt", map_location=self.device))
        self.q_network.load_state_dict(torch.load(f"{dir}/q_network.pt", map_location=self.device))


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
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 1024),
            nn.ReLU(),
            nn.Linear(1024, n_actions)
        )

    def forward(self, state):
        state = state.float() / 255.0
        state = state * 2 - 1  # Normalize

        return self.net(state)
