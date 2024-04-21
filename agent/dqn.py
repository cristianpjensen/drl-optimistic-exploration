import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from agent.agent import Agent
from agent.utils.scheduler import LinearScheduler


class AtariDQNAgent(Agent):
    def setup(self, config):
        # Ref: https://www.nature.com/articles/nature14236
        self.q_network = AtariValueNetwork(n_actions=config["n_actions"]).to(self.device)
        self.q_target = AtariValueNetwork(n_actions=config["n_actions"]).to(self.device).requires_grad_(False)
        self.q_target.load_state_dict(self.q_network.state_dict())

        # Params from https://www.nature.com/articles/nature14236
        self.optim = Adam(self.q_network.parameters(), lr=1e-4)
        self.scheduler = LinearScheduler([(0, 1), (config["train_steps"], 0.05)])
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
            q_values = self.q_network(state)

        action = np.zeros(state.shape[0], dtype=self.action_space.dtype)
        for i in range(state.shape[0]):
            if train:
                self.num_actions += 1

            if train and np.random.random() < self.scheduler.value(self.num_actions):
                action[i] = self.action_space.sample()
            else:
                action[i] = torch.argmax(q_values[i]).cpu().numpy()

            # Update target network every 10_000 actions
            if self.num_actions % self.target_update_freq == 0:
                self.q_target.load_state_dict(self.q_network.state_dict())

        return action

    def train(self, s_batch, a_batch, r_batch, s_next_batch, terminal_batch):
        # Compute target value
        with torch.no_grad():
            q_next_values = self.q_target(s_next_batch)
            q_next_values, _ = q_next_values.max(dim=1)
            target_q_values = r_batch + (1 - terminal_batch.float()) * self.gamma * q_next_values

        current_q_values = self.q_network(s_batch)
        current_q_values = torch.gather(current_q_values, dim=1, index=a_batch.unsqueeze(-1))
        current_q_values = current_q_values.squeeze(-1)

        loss = F.smooth_l1_loss(current_q_values, target_q_values)

        # Update weights
        self.optim.zero_grad()
        loss.backward()
        # Clip gradient norms
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
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

        # Input: 4 x 84 x 84

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
            nn.Linear(512, n_actions)
        )

    def forward(self, state):
        state = state.float() / 255.0
        return self.net(state)
