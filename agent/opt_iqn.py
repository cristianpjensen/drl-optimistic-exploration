"""

Dimension keys:

B: batch size
F: frame stack size
H: height
W: width
N: number of distribution samples for training values
T: number of distribution samples for target values
S: number of distribution samples for inference
A: number of available actions
M: embedding dimension
D: intermediate dimension between `conv` and `final_fc`

"""

import os
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import RMSprop

from agent.agent import Agent
from agent.utils.scheduler import LinearScheduler


class AtariOptIQNAgent(Agent):
    def setup(self, config):
        self.emb_dim = 64
        self.n_inf_samples = 32
        self.n_samples = 8
        self.n_target_samples = 8
        self.kappa = 1

        self.iqn_network = AtariIQNNetwork(
            n_actions=config["n_actions"],
            emb_dim=self.emb_dim,
        ).to(self.device)
        self.iqn_target = AtariIQNNetwork(
            n_actions=config["n_actions"],
            emb_dim=self.emb_dim,
        ).to(self.device).requires_grad_(False)
        self.iqn_target.load_state_dict(self.iqn_network.state_dict())

        self.optim = RMSprop(self.iqn_network.parameters(), lr=0.00025, alpha=0.95, eps=0.01)
        self.opt_scheduler = LinearScheduler([(0, 0.95), (2_000_000, 0.1), (10_000_000, 0.01)])
        self.gamma = config["gamma"]

        self.num_actions = 0
        self.num_updates = 0
        self.target_update_freq = 10_000

        # For logging the loss
        self.current_loss = 0
        self.logged_loss = True

    def act(self, state, train):
        with torch.no_grad():
            state_BFHW = torch.tensor(state, device=self.device)
            tau_BS = torch.rand((state_BFHW.shape[0], self.n_inf_samples), device=self.device)

            # Optimism with linear decay, combined with epsilon-greedy
            if train:
                opt_tau = self.opt_scheduler.value(self.num_actions)
                tau_BS = tau_BS * (1 - opt_tau) + opt_tau

            iq_values_BSA = self.iqn_network(state_BFHW, tau_BS)
            q_values_BA = iq_values_BSA.mean(dim=1)
           
        action = np.zeros(state.shape[0], dtype=self.action_space.dtype)
        for i in range(state.shape[0]):
            if train:
                self.num_actions += 1

            action[i] = torch.argmax(q_values_BA[i]).cpu().numpy()

            # Update target network every 10_000 actions
            if self.num_actions % self.target_update_freq == 0:
                self.iqn_target.load_state_dict(self.iqn_network.state_dict())

        return action

    def train(self, state_BFHW, action_B, reward_B, state_prime_BFHW, terminal_B):
        batch_size = state_BFHW.shape[0]

        # Compute target values
        with torch.no_grad():
            tau_BT = torch.rand((batch_size, self.n_target_samples), device=self.device)
            iq_next_BTA = self.iqn_target(state_prime_BFHW, tau_BT)
            iq_next_BT, _ = iq_next_BTA.max(dim=-1)
            target_BT = reward_B.unsqueeze(-1) + (1 - terminal_B.unsqueeze(-1).float()) * self.gamma * iq_next_BT

        tau_BN = torch.rand((batch_size, self.n_samples), device=self.device)
        iq_value_BNA = self.iqn_network(state_BFHW, tau_BN)
        iq_value_BN = iq_value_BNA[torch.arange(batch_size), :, action_B]

        # Compute Huber loss and TD error for every [tau, tau'] pair
        with warnings.catch_warnings(action="ignore"):
            # We want broadcasting here, so that we compute over all pairs
            huber_BNT = F.huber_loss(iq_value_BN.unsqueeze(2), target_BT.unsqueeze(1), reduction="none", delta=self.kappa)

        td_error_BNT = target_BT.unsqueeze(1) - iq_value_BN.unsqueeze(2)

        # Quantile regression loss for every [tau, tau'] pair
        loss_BNT = torch.abs(tau_BN.unsqueeze(2) - (td_error_BNT < 0).float()) * huber_BNT / self.kappa
        loss_B = loss_BNT.mean(dim=2).sum(dim=1)
        loss = loss_B.mean()
    
        # Update weights
        self.optim.zero_grad()
        loss.backward()
        # Clip gradient norms
        nn.utils.clip_grad_norm_(self.iqn_network.parameters(), 10)
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


class AtariIQNNetwork(nn.Module):
    """Implementation of IQN network, as in Bellemare book Chapter 10, and
    https://arxiv.org/pdf/1806.06923.pdf.

    Output: B x N x A

    """
    
    def __init__(self, n_actions: int, emb_dim: int):
        super(AtariIQNNetwork, self).__init__()

        self.emb_indices_M = nn.Parameter(
            torch.arange(emb_dim).float(),
            requires_grad=False,
        )
        self.n_actions = n_actions

        # Input: 4 x 84 x 84

        # DQN is `conv` followed by `final_fc`. IQN adds an embedding layer.
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),  # Output: 32 x 20 x 20
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # Output: 64 x 9 x 9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # Output: 64 x 7 x 7
            nn.Flatten()
        )

        self.embedding_fc = nn.Sequential(
            nn.Linear(emb_dim, 7 * 7 * 64),
            nn.ReLU()
        )

        self.final_fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, state_BFHW, tau_BN):
        state_BFHW = state_BFHW.float() / 255.0
        conv_out_BD = self.conv(state_BFHW)

        # Compute cosine embedding for each tau, resulting in N M-dimensional embeddings per state
        cos_embedding_BNM = torch.cos(torch.pi * torch.einsum("bn,m->bnm", tau_BN, self.emb_indices_M))
        emb_out_BND = self.embedding_fc(cos_embedding_BNM)

        # Hadamard product between conv_out and emb_out for each tau
        hadamard = torch.einsum("bd,bnd->bnd", conv_out_BD, emb_out_BND)

        return self.final_fc(hadamard)
