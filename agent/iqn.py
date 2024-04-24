"""
Dimension suffix keys:

B: batch size
A: number of available actions
m: number of samples from the distributions to choose action
M: size of cosine embedding
n: number of percentile samples for current states
n_prime: number of percentile samples for next states


!!!!!!!!!!!!!!!!!IMPORTANT!!!!!!!!!!!!!!!!!!!!!!!
the paper says that a larger n yields better results. The complexity
seems to grow quadratically with n. the paper also says that after n = 8
we see diminishing returns and that 8 is enough to observe improvements
over C51/QR. Problem: with n = 8 I have 5 iter/s. With n = 4 20 iter/s
n = 2 yield 40ish iter per second, and with n = 1 you are expected to recover DQN.

"""


import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from agent.agent import Agent
from agent.utils.scheduler import LinearScheduler


class AtariIQNAgent(Agent):
    def setup(self, config):
        self.M = 64
        self.m = 32
        self.n = 2
        self.n_prime = self.n
        self.k = 1

        self.iqn_network = AtariIQNNetwork(
            n_actions=config["n_actions"],
            M=self.M,
        ).to(self.device)
        self.iqn_target = AtariIQNNetwork(
            n_actions=config["n_actions"],
            M=self.M,
        ).to(self.device).requires_grad_(False)
        self.iqn_target.load_state_dict(self.iqn_network.state_dict())

        self.optim = Adam(self.iqn_network.parameters(), lr=0.00025)
        
        # Linearly interpolate between 0 and 10% of number of training steps
        self.scheduler = LinearScheduler([(0, 1), (config["train_steps"] // 10, 0.05)])
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
           
        action = np.zeros(state.shape[0], dtype=self.action_space.dtype)
        for i in range(state.shape[0]):
            if train:
                self.num_actions += 1

            if train and np.random.random() < self.scheduler.value(self.num_actions):
                action[i] = self.action_space.sample()
            else:
                #what happens here is that for each of the 4 states, we compute a sample average 
                #(m is the number of samples used, value taken from iqn paper)
                #for the value of each action for that state, and than we take the action
                # with the highest
                #average. Basic approach without optimism.
                #maybe m is the bottleneck on why it's so slow
                 
                taus = torch.rand(self.m)
                q_values = torch.zeros(self.m, self.action_space.n)
                
                for j, tau in enumerate(taus):
                    q_values[j] = self.iqn_network(state[i], tau)
                    
                avg_q_values = torch.mean(q_values, dim = 0)
                action[i] = torch.argmax(avg_q_values).cpu().numpy()
                

            # Update target network every 10_000 actions
            if self.num_actions % self.target_update_freq == 0:
                self.iqn_target.load_state_dict(self.iqn_network.state_dict())

        return action

    def train(self, s_batch, a_batch, r_batch, s_next_batch, terminal_batch):
        
        #the IQN paper says that the higher n, n_prime the better
        #but also that after n = 8 there are diminishing returns
        
        n = self.n
        n_prime = self.n_prime

        batch_size = s_batch.size(0)

        taus = torch.rand(n, batch_size)
        taus_prime = torch.rand(n_prime, batch_size)

        q_values_mat = torch.zeros(n, batch_size)

        for i in range(n):
            q_values = self.iqn_network(s_batch, taus[i])
            q_value = q_values[torch.arange(q_values.shape[0]).long(), a_batch.long()]
            q_values_mat[i] = q_value

        q_values_mat_transposed = q_values_mat.transpose(0,1)
        q_next_values_mat = torch.zeros(n_prime, batch_size)

        for i in range(n_prime):
            q_next_values = self.iqn_target(s_next_batch, taus_prime[i])
            q_next_values_mat[i] = q_next_values.max(1).values

        q_next_values_mat_transposed = q_values_mat.transpose(0,1)


        huberloss = torch.nn.HuberLoss(delta = self.k)
        indicator = lambda u : ( u < 0).float()

        loss = 0

        for batch_index in range(batch_size):

            for tau_index, q_value in enumerate(q_values_mat_transposed[batch_index]):
                for q_next_value in q_next_values_mat_transposed[batch_index]:
                    
                    target = r_batch[batch_index] + (self.gamma * q_next_value) * (1 - terminal_batch[batch_index].float())
                    
                    error = target - q_value
                    
                    loss += torch.abs(indicator(error) - taus[tau_index, batch_index] ) * huberloss( q_value, target)
            loss = loss / n_prime

        loss = loss / batch_size
    
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
    https://arxiv.org/pdf/1806.06923.pdf."""
    
    def __init__(self, n_actions: int, emb_dim: int):
        super(AtariIQNNetwork, self).__init__()

        self.emb_indices_M = torch.arange(emb_dim).float()
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
        """
        Dimension keys:

        B: batch size
        F: frame stack size
        H: height
        W: width
        A: number of available actions
        M: size of cosine embedding
        N: number of distribution samples per state
        D: intermediate dimension between `conv` and `final_fc`

        Output: B x N x A

        """

        state_BFHW = state_BFHW.float() / 255.0
        conv_out_BD = self.conv(state_BFHW)

        # Compute cosine embedding for each tau, resulting in N M-dimensional embeddings per state
        cos_embedding_BNM = torch.cos(torch.pi * torch.einsum("bn,m->bnm", tau_BN, self.emb_indices_M))
        emb_out_BND = self.embedding_fc(cos_embedding_BNM)

        # Hadamard product between conv_out and emb_out for each tau
        hadamard = torch.einsum("bd,bnd->bnd", conv_out_BD, emb_out_BND)

        return self.final_fc(hadamard)
