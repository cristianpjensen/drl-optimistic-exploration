import numpy as np
import torch

from agent.agent import Agent
from agent.utils.scheduler import LinearScheduler


class TabularQLearningAgent(Agent):
    """An agent that selects actions uniformly at random."""

    def setup(self, config):
        # Q-values for each state-action pair
        self.q_values = torch.zeros((config["n_states"], config["n_actions"]))
        self.lr = 0.01
        self.gamma = config["gamma"]

        self.scheduler = LinearScheduler([(0, 1), (50_000, 0.1)])

        self.num_actions = 0
        self.loss = 0

    def act(self, state, train):
        state = torch.from_numpy(state).argmax(dim=-1).squeeze(1)
        q_values = self.q_values[state]

        action = np.zeros(state.shape[0], dtype=self.action_space.dtype)
        for i in range(state.shape[0]):
            self.num_actions += 1

            if train and np.random.random() < self.scheduler.value(self.num_actions):
                action[i] = self.action_space.sample()
            else:
                action[i] = torch.argmax(q_values[i]).cpu().numpy()

        return action

    def train(self, s_batch, a_batch, r_batch, s_next_batch, terminal_batch):
        s_batch = s_batch.argmax(dim=-1).squeeze(1)
        s_next_batch = s_next_batch.argmax(dim=-1).squeeze(1)

        q_values = self.q_values[s_batch]
        q_value = q_values[range(q_values.shape[0]), a_batch.long()]
        q_next_value = self.q_values[s_next_batch].max(1).values

        temporal_difference = r_batch + self.gamma * q_next_value * (1 - terminal_batch.float()) - q_value

        # Update Q values
        self.q_values[s_batch, a_batch] += self.lr * temporal_difference

        self.loss = temporal_difference.mean()

    def log(self, run):
        run["train/loss"].append(self.loss)

    def save(self, dir):
        pass

    def load(self, dir):
        pass
