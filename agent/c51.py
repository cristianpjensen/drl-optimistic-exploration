"""
Dimension keys:

B: batch size
F: frame stack size
H: height
W: width
N: number of distribution samples for training values
T: number of distribution samples for target values
K: number of distribution samples for inference
A: number of available actions
M: embedding dimension
D: intermediate dimension of DQN
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from agent.agent import Agent
from agent.utils.loss import quantile_huber_loss
from agent.utils.scheduler import LinearScheduler
from agent.utils.disable_gradients import disable_gradients
from agent.networks.dqn import AtariDQNFeatures, QNetwork


class AtariIQNAgent(Agent):
    def setup(self, config):
        self.n_categories = 51
        self.v_min = -10
        self.v_max = 10
        self.values_N = torch.linspace(self.v_min, self.v_max, self.n_categories, device=self.device)
        self.delta_z = (self.v_max - self.v_min) / (self.n_categories - 1)

        self.cat_network = AtariC51Network(config["n_actions"], self.n_categories).to(self.device)
        self.cat_target = AtariC51Network(config["n_actions"], self.n_categories).to(self.device)
        self.cat_target.load_state_dict(self.cat_network.state_dict())
        disable_gradients(self.cat_target)

        self.optim = Adam(
            self.cat_network.parameters(),
            lr=0.00025,
            eps=0.01 / config["batch_size"]
        )
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
                probs_BAN = self.cat_network(state_BFHW)

            q_values_BA = torch.einsum("ban,n->ba", probs_BAN, self.values_N)
            actions_B = torch.argmax(q_values_BA, dim=1).cpu().numpy()

        if train:
            self.num_actions += state.shape[0]

            if self.num_actions % self.target_update_freq < state.shape[0]:
                self.iqn_target.load_state_dict(self.iqn_network.state_dict())

        return actions_B

    def train(self, state_BFHW, action_B, reward_B, state_prime_BFHW, terminal_B):
        batch_size = state_BFHW.shape[0]

        # Compute target distributions
        with torch.no_grad():
            probs_prime_BAN = self.cat_target(state_prime_BFHW)
            q_prime_BA = torch.einsum("ban,n->ba", probs_prime_BAN, self.values_N)
            action_star_B = torch.argmax(q_prime_BA, dim=1)

            target_probs_BAN = probs_prime_BAN[torch.arange(batch_size), :, action_star_B]
            target_probs_BAN = reward_B.unsqueeze(-1) + (1 - terminal_B.unsqueeze(-1).float()) * self.gamma * target_probs_BAN

        # # Compute target values
        # with torch.no_grad():
        #     tau_BT = torch.rand((batch_size, self.n_target_samples), device=self.device)
        #     iq_next_BTA = self.iqn_target(state_prime_BFHW, tau_BT)

        #     # Get the best next action
        #     q_next_BA = iq_next_BTA.mean(dim=1)
        #     action_star_B = torch.argmax(q_next_BA, dim=1)

        #     # Compute target values using the best next action
        #     q_action_star_BT = iq_next_BTA[torch.arange(batch_size), :, action_star_B]
        #     target_BT = reward_B.unsqueeze(-1) + (1 - terminal_B.unsqueeze(-1).float()) * self.gamma * q_action_star_BT

        # tau_BN = torch.rand((batch_size, self.n_samples), device=self.device)
        # iq_value_BNA = self.iqn_network(state_BFHW, tau_BN)
        # iq_value_BN = iq_value_BNA[torch.arange(batch_size), :, action_B]

        # td_error_BNT = target_BT.unsqueeze(1) - iq_value_BN.unsqueeze(2)
        # loss = quantile_huber_loss(td_error_BNT, tau_BN, self.kappa)
        # loss = loss.mean()

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
        torch.save(self.cat_network.state_dict(), f"{dir}/cat_network.pt")
        return True

    def load(self, dir):
        self.cat_network.load_state_dict(torch.load(f"{dir}/cat_network.pt", map_location=self.device))
        self.cat_target.load_state_dict(torch.load(f"{dir}/cat_network.pt", map_location=self.device))


class AtariC51Network(nn.Module):
    """Implementation of C51 network.

    Dimension keys:
        B: batch size
        A: number of available actions
        N: number of categories

    Output: [B, A, N]

    """
    
    def __init__(self, n_actions: int, n_categories: int):
        super(AtariC51Network, self).__init__()

        self.n_actions = n_actions
        self.n_categories = n_categories

        self.net = nn.Sequential(
            AtariDQNFeatures(),
            QNetwork(3136, 512, n_actions * n_categories),
        )

    def forward(self, state_BFHW):
        log_probs_BAN = self.net(state_BFHW).view(-1, self.n_actions, self.n_categories)
        return F.softmax(log_probs_BAN, dim=2)
