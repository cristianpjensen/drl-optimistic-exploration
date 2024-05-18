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
from agent.utils.disable_gradients import disable_gradients
from agent.utils.scheduler import LinearScheduler
from agent.networks.bayesian_dqn import BayesianAtariDQNFeatures, BayesianQNetwork


class AtariBayesianOptC51Agent(Agent):
    def setup(self, config):
        self.n_categories = 51
        self.v_min = -10
        self.v_max = 10
        self.values_N = torch.linspace(self.v_min, self.v_max, self.n_categories, device=self.device)
        self.delta_z = (self.v_max - self.v_min) / (self.n_categories - 1)

        self.cat_network = BayesianAtariC51Network(config["n_actions"], self.n_categories).to(self.device)
        self.cat_target = BayesianAtariC51Network(config["n_actions"], self.n_categories).to(self.device)
        self.cat_target.load_state_dict(self.cat_network.state_dict())
        disable_gradients(self.cat_target)

        self.optim = Adam(self.cat_network.parameters(), lr=0.00025, eps=0.01 / config["batch_size"])
        self.scheduler = LinearScheduler([(0, 0), (0, 0)])
        self.opt_scheduler = LinearScheduler([(0, 0.5), (5_000_000, 0.1), (20_000_000, 0.01)])
        self.gamma = config["gamma"]

        self.num_actions = 0
        self.num_updates = 0
        self.target_update_freq = 10_000

        # For logging the loss
        self.current_loss = [0, 0, 0]
        self.logged_loss = True

    def act(self, state, train):
        if train and np.random.random() < self.scheduler.value(self.num_actions):
            actions_B = np.zeros(state.shape[0], dtype=self.action_space.dtype)
            for i in range(state.shape[0]):
                actions_B[i] = self.action_space.sample()
        else:
            with torch.no_grad():
                state_BFHW = torch.tensor(state, device=self.device)
                probs_BAN = self.cat_network(state_BFHW, train)

            # Optimistic sampling
            if train:
                opt_tau = self.opt_scheduler.value(self.num_actions)
                cdf_BAN = probs_BAN.cumsum(dim=2)
                probs_BAN = torch.where(cdf_BAN >= opt_tau, probs_BAN, torch.zeros_like(probs_BAN))

            q_values_BA = torch.sum(probs_BAN * self.values_N, dim=2)
            actions_B = torch.argmax(q_values_BA, dim=1).cpu().numpy()

        if train:
            self.num_actions += state.shape[0]

            if self.num_actions % self.target_update_freq < state.shape[0]:
                self.cat_target.load_state_dict(self.cat_network.state_dict())

        return actions_B

    def train(self, state_BFHW, action_B, reward_B, state_prime_BFHW, terminal_B):
        batch_size = state_BFHW.shape[0]

        # Compute target value
        with torch.no_grad():
            probs_prime_BAN = self.cat_target(state_prime_BFHW, train=False)
            q_prime_BA = torch.sum(probs_prime_BAN * self.values_N, dim=2)
            action_star_B = torch.argmax(q_prime_BA, dim=1)

            # Temp: next_dist
            probs_prime_BN = probs_prime_BAN[torch.arange(batch_size), action_star_B]
            # Tz
            target_values_BN = reward_B.unsqueeze(-1) + (1 - terminal_B.unsqueeze(-1).float()) * self.gamma * self.values_N
            target_values_BN = torch.clamp(target_values_BN, self.v_min, self.v_max)

        b = (target_values_BN - self.v_min) / self.delta_z
        lower = torch.floor(b)
        upper = torch.ceil(b)

        target_probs_BN = torch.zeros((batch_size, self.n_categories), device=self.device)
        target_probs_BN = target_probs_BN.scatter_add(1, lower.long(), probs_prime_BN * (upper - b))
        target_probs_BN = target_probs_BN.scatter_add(1, upper.long(), probs_prime_BN * (b - lower))

        probs_BAN = self.cat_network(state_BFHW, train=True)
        probs_BN = probs_BAN[torch.arange(batch_size), action_B]

        td_loss = -torch.sum(target_probs_BN * torch.log(probs_BN + 1e-5), dim=1)
        td_loss = td_loss.mean()
        
        kl_loss = self.cat_network.prior_kl_loss()
        loss = td_loss + 0.1 * kl_loss

        # Update weights
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.num_updates += 1
        self.current_loss = [loss, td_loss, kl_loss]
        self.logged_loss = False

    def log(self, run):
        if not self.logged_loss:
            run["train/loss"].append(self.current_loss[0])
            run["train/td_loss"].append(self.current_loss[1])
            run["train/kl_loss"].append(self.current_loss[2])
            self.logged_loss = True

        run["train/num_actions"].append(self.num_actions)
        run["train/num_updates"].append(self.num_updates)

    def save(self, dir) -> bool:
        os.makedirs(dir, exist_ok=True)
        torch.save(self.cat_network.state_dict(), f"{dir}/cat_network.pt")
        return True

    def load(self, dir):
        self.cat_target.load_state_dict(torch.load(f"{dir}/cat_network.pt", map_location=self.device))
        self.cat_network.load_state_dict(torch.load(f"{dir}/cat_network.pt", map_location=self.device))


class BayesianAtariC51Network(nn.Module):
    """Bayesian categorical Q-network for Atari environments.

    Ref: https://arxiv.org/abs/1707.06887

    Dimension keys:
        B: batch size
        A: number of available actions
        N: number of categories in the distribution

    Output: [B, A, N]
    
    """

    def __init__(self, n_actions: int, n_categories: int):
        super(BayesianAtariC51Network, self).__init__()

        self.n_actions = n_actions
        self.n_categories = n_categories

        self.features = BayesianAtariDQNFeatures()
        self.q_network = BayesianQNetwork(3136, 512, n_actions * n_categories)

    def forward(self, state, train=True):
        feats = self.features(state, train)
        log_probs_BAN = self.q_network(feats, train).view(-1, self.n_actions, self.n_categories)
        return F.softmax(log_probs_BAN, dim=2)

    def prior_kl_loss(self):
        return self.features.prior_kl_loss() + self.q_network.prior_kl_loss()
