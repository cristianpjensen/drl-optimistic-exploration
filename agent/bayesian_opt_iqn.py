import os

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from agent.agent import Agent
from agent.networks.bayesian_dqn import BayesianAtariDQNFeatures, BayesianQNetwork
from agent.networks.cosine_embedding import CosineEmbedding
from agent.utils.disable_gradients import disable_gradients
from agent.utils.loss import quantile_huber_loss
from agent.utils.scheduler import LinearScheduler


class AtariBayesianOptIQNAgent(Agent):
    def setup(self, config):
        self.emb_dim = 64
        self.n_inf_samples = 32
        self.n_samples = 8
        self.n_target_samples = 8
        self.kappa = 1

        self.iqn_network = BayesianAtariIQNNetwork(config["n_actions"]).to(self.device)
        self.iqn_target = BayesianAtariIQNNetwork(config["n_actions"]).to(self.device)
        self.iqn_target.load_state_dict(self.iqn_network.state_dict())
        disable_gradients(self.iqn_target)

        self.optim = Adam(
            self.iqn_network.parameters(),
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
        # Epsilon-greedy
        if train and np.random.random() < self.scheduler.value(self.num_actions):
            actions_B = np.zeros(state.shape[0], dtype=self.action_space.dtype)
            for i in range(state.shape[0]):
                actions_B[i] = self.action_space.sample()
        else:
            with torch.no_grad():
                state_BFHW = torch.tensor(state, device=self.device)
                tau_BK = torch.rand((state_BFHW.shape[0], self.n_inf_samples), device=self.device)

                # Optimistic sampling
                if train:
                    opt_tau = self.opt_scheduler.value(self.num_actions)
                    tau_BK = opt_tau + tau_BK * (1 - opt_tau)

                iq_values_BKA = self.iqn_network(state_BFHW, tau_BK, train)

            q_values_BA = torch.mean(iq_values_BKA, dim=1)
            actions_B = torch.argmax(q_values_BA, dim=1).cpu().numpy()

        if train:
            self.num_actions += state.shape[0]

            if self.num_actions % self.target_update_freq < state.shape[0]:
                self.iqn_target.load_state_dict(self.iqn_network.state_dict())

        return actions_B

    def train(self, state_BFHW, action_B, reward_B, state_prime_BFHW, terminal_B):
        batch_size = state_BFHW.shape[0]

        # Compute target values
        with torch.no_grad():
            tau_BT = torch.rand((batch_size, self.n_target_samples), device=self.device)
            iq_next_BTA = self.iqn_target(state_prime_BFHW, tau_BT, train=False)

            # Get the best next action
            q_next_BA = iq_next_BTA.mean(dim=1)
            action_star_B = torch.argmax(q_next_BA, dim=1)

            # Compute target values using the best next action
            q_action_star_BT = iq_next_BTA[torch.arange(batch_size), :, action_star_B]
            target_BT = reward_B.unsqueeze(-1) + (1 - terminal_B.unsqueeze(-1).float()) * self.gamma * q_action_star_BT

        tau_BN = torch.rand((batch_size, self.n_samples), device=self.device)
        iq_value_BNA = self.iqn_network(state_BFHW, tau_BN, train=True)
        iq_value_BN = iq_value_BNA[torch.arange(batch_size), :, action_B]

        td_error_BNT = target_BT.unsqueeze(1) - iq_value_BN.unsqueeze(2)
        td_loss = quantile_huber_loss(td_error_BNT, tau_BN, self.kappa)
        td_loss = td_loss.mean()

        kl_loss = self.iqn_network.prior_kl_loss()
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
        torch.save(self.iqn_network.state_dict(), f"{dir}/iqn_network.pt")
        return True

    def load(self, dir):
        self.iqn_target.load_state_dict(torch.load(f"{dir}/iqn_network.pt", map_location=self.device))
        self.iqn_network.load_state_dict(torch.load(f"{dir}/iqn_network.pt", map_location=self.device))


class BayesianAtariIQNNetwork(nn.Module):
    """Bayesian implementation of IQN network, as in Bellemare book Chapter 10, and
    https://arxiv.org/pdf/1806.06923.pdf.

    Dimension keys:
        B: batch size
        N: number of distribution samples
        A: number of available actions

    Output: [B, N, A]

    """
    
    def __init__(self, n_actions: int):
        super(BayesianAtariIQNNetwork, self).__init__()

        self.feat_net = BayesianAtariDQNFeatures()
        self.emb_net = CosineEmbedding(64)
        self.q_net = BayesianQNetwork(3136, 512, n_actions)

    def forward(self, state_BFHW, tau_BN, train=True):
        feat_BD = self.feat_net(state_BFHW, train)
        emb_BND = self.emb_net(tau_BN)
        hadamard_BND = feat_BD.unsqueeze(1) * emb_BND
        return self.q_net(hadamard_BND, train)

    def prior_kl_loss(self):
        return self.feat_net.prior_kl_loss() + self.q_net.prior_kl_loss()
