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
from torch.optim import RMSprop

from agent.agent import Agent
from agent.utils.huber_loss import huber_loss
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

    def train(self, state_BFHW, action_B, reward_B, state_prime_BFHW, terminal_B):
        batch_size = state_BFHW.shape[0]

        # Comute distributional Bellman target
        with torch.no_grad():
            qr_next_BAQ = self.qr_target(state_prime_BFHW)
            qr_next_BA = qr_next_BAQ.mean(dim=2)
            action_star_B = torch.argmax(qr_next_BA, dim=1)

            quantile_action_star_BQ = qr_next_BAQ[torch.arange(batch_size), action_star_B]
            target_BQ = reward_B.unsqueeze(-1) + (1 - terminal_B.unsqueeze(-1).float()) * self.gamma * quantile_action_star_BQ

        # Compute quantile regression loss
        qr_value_BAQ = self.qr_network(state_BFHW)
        qr_value_BQ = qr_value_BAQ[torch.arange(batch_size), action_B]

        # The tensor dimensions are [batch, value, target]
        td_error_BQQ = target_BQ.unsqueeze(1) - qr_value_BQ.unsqueeze(2)
        huber_BQQ = huber_loss(td_error_BQQ, self.kappa)

        # Quantile Huber loss
        loss_BQQ = torch.abs(self.tau_Q - (td_error_BQQ < 0).float()) * huber_BQQ
        loss_B = loss_BQQ.mean(dim=2).sum(dim=1)
        loss = loss_B.mean()

        # Update weights
        self.optim.zero_grad()
        loss.backward()
        # Clip gradient norms
        nn.utils.clip_grad_norm_(self.qr_network.parameters(), 10)
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
