"""
Dimension keys:

B: batch size
A: number of available actions
Q: number of quantiles
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import RMSprop

from agent.agent import Agent
from agent.utils.scheduler import LinearScheduler


class AtariQRAgent(Agent):
    def setup(self, config):
        self.n_quantiles = 51

        self.qr_network = AtariQRNetwork(
            n_actions=config["n_actions"],
            n_quantiles=self.n_quantiles,
        ).to(self.device)
        self.qr_target = AtariQRNetwork(
            n_actions=config["n_actions"],
            n_quantiles=self.n_quantiles,
        ).to(self.device).requires_grad_(False)
        self.qr_target.load_state_dict(self.qr_network.state_dict())

        self.tau_Q = (torch.arange(self.n_quantiles, device=self.device) * 2 - 1) / (2 * self.n_quantiles)
        self.kappa = 1

        self.optim = RMSprop(self.qr_network.parameters(), lr=0.00025, alpha=0.95, eps=0.01)
        self.scheduler = LinearScheduler([(0, 1), (1000000, 0.01)])
        self.gamma = config["gamma"]

        self.num_actions = 0
        self.num_updates = 0
        self.target_update_freq = 10_000

        # For logging the loss
        self.current_loss = 0
        self.logged_loss = True

    def act(self, state, train):
        with torch.no_grad():
            state = torch.tensor(state, device=self.device)
            qr_values_BAQ = self.qr_network(state)
            q_values_BA = qr_values_BAQ.mean(dim=-1)

        action = np.zeros(state.shape[0], dtype=self.action_space.dtype)
        for i in range(state.shape[0]):
            if train:
                self.num_actions += 1

            if train and np.random.random() < self.scheduler.value(self.num_actions):
                action[i] = self.action_space.sample()
            else:
                action[i] = torch.argmax(q_values_BA[i]).cpu().numpy()

            # Update target network every 10_000 actions
            if self.num_actions % self.target_update_freq == 0:
                self.qr_target.load_state_dict(self.qr_network.state_dict())

        return action

    def train(self, s_batch, a_batch, r_batch, s_next_batch, terminal_batch):
        # Compute target value
        with torch.no_grad():
            qr_next_values_BAQ = self.qr_target(s_next_batch)
            q_values_BA = qr_next_values_BAQ.mean(dim=-1)
            q_target_B, _ = q_values_BA.max(dim=-1)
            target_q_values_B = r_batch + (1 - terminal_batch.float()) * self.gamma * q_target_B

        qr_values_BAQ = self.qr_network(s_batch)
        qr_values_BQ = qr_values_BAQ[torch.arange(qr_values_BAQ.shape[0]), a_batch]

        loss = self._loss(qr_values_BQ, target_q_values_B)
        loss = loss.mean()

        # Update weights
        self.optim.zero_grad()
        loss.backward()
        # Clip gradient norms
        nn.utils.clip_grad_norm_(self.qr_network.parameters(), 10)
        self.optim.step()

        self.num_updates += 1
        self.current_loss = loss
        self.logged_loss = False

    def _loss(self, input_BQ: torch.Tensor, target_B: torch.Tensor):
        target_BQ = target_B.unsqueeze(-1).expand_as(input_BQ)
        u = target_BQ - input_BQ
        return torch.abs(self.tau_Q - (u < 0).float()) * F.huber_loss(input_BQ, target_BQ, delta=self.kappa)

    def log(self, run):
        if not self.logged_loss:
            run["train/loss"].append(torch.mean(self.current_loss))
            self.logged_loss = True

        run["train/num_actions"].append(self.num_actions)
        run["train/num_updates"].append(self.num_updates)

    def save(self, dir) -> bool:
        os.makedirs(dir, exist_ok=True)
        torch.save(self.qr_network.state_dict(), f"{dir}/qr_network.pt")
        return True

    def load(self, dir):
        self.qr_target.load_state_dict(torch.load(f"{dir}/qr_network.pt", map_location=self.device))
        self.qr_network.load_state_dict(torch.load(f"{dir}/qr_network.pt", map_location=self.device))


class AtariQRNetwork(nn.Module):
    """ Ref: https://arxiv.org/pdf/1710.10044.pdf """

    def __init__(self, n_actions: int, n_quantiles: int):
        super(AtariQRNetwork, self).__init__()

        # Input: 4 x 84 x 84

        self.n_quantiles = n_quantiles
        self.n_actions = n_actions

        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),  # Output: 32 x 20 x 20
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # Output: 64 x 9 x 9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # Output: 64 x 7 x 7
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions * n_quantiles)
        )

    def forward(self, state):
        state = state.float() / 255.0
        output = self.net(state)
        return output.view(output.shape[0], self.n_actions, self.n_quantiles)
