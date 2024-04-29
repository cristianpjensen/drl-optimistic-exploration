"""
Dimension keys:

B: batch size
A: number of available actions
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from agent.agent import Agent
from agent.utils.scheduler import LinearScheduler
from agent.networks.dqn import AtariDQNFeatures, QNetwork
from agent.utils.disable_gradients import disable_gradients


class AtariDQNAgent(Agent):
    def setup(self, config):
        # Ref: https://www.nature.com/articles/nature14236
        self.q_network = AtariValueNetwork(config["n_actions"]).to(self.device)
        self.q_target = AtariValueNetwork(config["n_actions"]).to(self.device)
        self.q_target.load_state_dict(self.q_network.state_dict())
        disable_gradients(self.q_target)

        self.optim = Adam(self.q_network.parameters(), lr=0.00025, eps=0.01 / config["batch_size"])
        self.scheduler = LinearScheduler([(0, 1), (1000000, 0.01)])
        self.gamma = config["gamma"]

        self.num_actions = 0
        self.num_updates = 0
        self.target_update_freq = 10_000

        # For logging the loss
        self.current_loss = 0
        self.logged_loss = True

    def act(self, state, train):
        if train and np.random.random() < self.scheduler.value(self.num_actions):
            actions_B = np.zeros(state.shape[0], dtype=self.action_space.dtype)
            for i in range(state.shape[0]):
                actions_B[i] = self.action_space.sample()
        else:
            with torch.no_grad():
                state_BFHW = torch.tensor(state, device=self.device)
                q_values_BA = self.q_network(state_BFHW)

            actions_B = torch.argmax(q_values_BA, dim=1).cpu().numpy()

        if train:
            self.num_actions += state.shape[0]

            if self.num_actions % self.target_update_freq < state.shape[0]:
                self.q_target.load_state_dict(self.q_network.state_dict())

        return actions_B

    def train(self, s_batch, a_batch, r_batch, s_next_batch, terminal_batch):
        # Compute target value
        with torch.no_grad():
            q_next_values_BA = self.q_target(s_next_batch)
            q_next_values_B, _ = q_next_values_BA.max(dim=1)
            target_q_values_B = r_batch + (1 - terminal_batch.float()) * self.gamma * q_next_values_B

        q_values_BA = self.q_network(s_batch)
        q_values_B = q_values_BA[torch.arange(q_values_BA.shape[0]), a_batch]

        loss = F.smooth_l1_loss(q_values_B, target_q_values_B)

        # Update weights
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.num_updates += 1
        self.current_loss = loss
        self.logged_loss = False

    def log(self, run):
        if not self.logged_loss:
            run["train/loss"].append(torch.mean(self.current_loss))
            self.logged_loss = True

        run["train/num_actions"].append(self.num_actions)
        run["train/num_updates"].append(self.num_updates)

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

        self.net = nn.Sequential(
            AtariDQNFeatures(),
            QNetwork(3136, 512, n_actions),
        )

    def forward(self, state):
        return self.net(state)
