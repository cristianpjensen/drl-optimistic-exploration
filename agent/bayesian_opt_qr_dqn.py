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
from torch.optim import Adam

from agent.agent import Agent
from agent.networks.bayesian_dqn import BayesianAtariDQNFeatures, BayesianQNetwork
from agent.utils.loss import quantile_huber_loss
from agent.utils.scheduler import LinearScheduler


class AtariBayesianOptQRAgent(Agent):
    def setup(self, config):
        self.n_quantiles = 200

        self.qr_network = BayesianAtariQRNetwork(
            n_actions=config["n_actions"],
            n_quantiles=self.n_quantiles,
        ).to(self.device)
        self.qr_target = BayesianAtariQRNetwork(
            n_actions=config["n_actions"],
            n_quantiles=self.n_quantiles,
        ).to(self.device).requires_grad_(False)
        self.qr_target.load_state_dict(self.qr_network.state_dict())

        self.tau_Q = (torch.arange(1, self.n_quantiles + 1, device=self.device) * 2 - 1) / (2 * self.n_quantiles)
        self.kappa = 1

        self.optim = Adam(
            self.qr_network.parameters(),
            lr=0.00025,
            eps=0.01 / config["batch_size"],
        )
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
                qr_values_BQA = self.qr_network(state_BFHW, train=train)

            # Optimistic sampling by zeroing out quantiles below the optimistic tau
            if train:
                opt_tau = self.opt_scheduler.value(self.num_actions)
                n_taus = (self.tau_Q >= opt_tau).float().sum()
                qr_values_BQA = torch.where(self.tau_Q[None, :, None] >= opt_tau, qr_values_BQA, torch.zeros_like(qr_values_BQA))
                q_values_BA = torch.sum(qr_values_BQA, dim=1) / n_taus
            else:
                q_values_BA = torch.mean(qr_values_BQA, dim=1)

            actions_B = torch.argmax(q_values_BA, dim=1).cpu().numpy()

        if train:
            self.num_actions += state.shape[0]

            if self.num_actions % self.target_update_freq < state.shape[0]:
                self.qr_target.load_state_dict(self.qr_network.state_dict())

        return actions_B

    def train(self, state_BFHW, action_B, reward_B, state_prime_BFHW, terminal_B):
        batch_size = state_BFHW.shape[0]

        # Compute distributional Bellman target
        with torch.no_grad():
            qr_next_BQA = self.qr_target(state_prime_BFHW, train=False)
            q_next_BA = qr_next_BQA.mean(dim=1)
            action_star_B = torch.argmax(q_next_BA, dim=1)

            q_action_star_BQ = qr_next_BQA[torch.arange(batch_size), :, action_star_B]
            target_BQ = reward_B.unsqueeze(-1) + (1 - terminal_B.unsqueeze(-1).float()) * self.gamma * q_action_star_BQ

        # Compute quantile regression loss
        qr_value_BQA = self.qr_network(state_BFHW, train=True)
        qr_value_BQ = qr_value_BQA[torch.arange(batch_size), :, action_B]

        # The tensor dimensions are [batch, value, target]
        td_error_BQQ = target_BQ.unsqueeze(1) - qr_value_BQ.unsqueeze(2)
        td_loss = quantile_huber_loss(td_error_BQQ, self.tau_Q.unsqueeze(0), self.kappa)
        td_loss = td_loss.mean()

        kl_loss = self.qr_network.prior_kl_loss()
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
        torch.save(self.qr_network.state_dict(), f"{dir}/qr_network.pt")
        return True

    def load(self, dir):
        self.qr_target.load_state_dict(torch.load(f"{dir}/qr_network.pt", map_location=self.device))
        self.qr_network.load_state_dict(torch.load(f"{dir}/qr_network.pt", map_location=self.device))


class BayesianAtariQRNetwork(nn.Module):
    """Outputs Q quantile values for each action using Bayesian layers.

    Ref: https://arxiv.org/pdf/1710.10044.pdf

    Dimension keys:
        B: batch size
        A: number of available actions
        Q: number of quantiles

    Output: [B, Q, A]
    
    """

    def __init__(self, n_actions: int, n_quantiles: int):
        super(BayesianAtariQRNetwork, self).__init__()

        self.n_quantiles = n_quantiles
        self.n_actions = n_actions

        self.feat_net = BayesianAtariDQNFeatures()
        self.q_net = BayesianQNetwork(3136, 512, n_quantiles * n_actions)

    def forward(self, state, train=True):
        feats = self.feat_net(state, train)
        return self.q_net(feats, train).view(state.shape[0], self.n_quantiles, self.n_actions)

    def prior_kl_loss(self):
        return self.feat_net.prior_kl_loss() + self.q_net.prior_kl_loss()
