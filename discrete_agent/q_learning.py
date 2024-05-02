from agent.utils.scheduler import LinearScheduler
from discrete_agent.discrete_agent import DiscreteAgent
import numpy as np


class QLearning(DiscreteAgent):
    def setup(self, config):
        self.q_values_SA = np.zeros((config["n_states"], config["n_actions"]))
        self.alpha = 0.1
        self.gamma = config["gamma"]
        self.scheduler = LinearScheduler([(0, 1), (50_000, 0)])
        self.num_actions = 0

        self.loss = 0
        self.logged_loss = False

    def act(self, state, train):
        if train:
            self.num_actions += 1
            if np.random.random() < self.scheduler.value(self.num_actions):
                return self.action_space.sample()
            else:
                return np.argmax(self.q_values_SA[state])

    def update_policy(self, state, action, reward, next_state, terminal):
        # Compute temporal difference
        q_current = self.q_values_SA[state, action]
        q_next = np.max(self.q_values_SA[next_state])
        target = reward + self.gamma * q_next * (1-terminal)
        td_error = target - q_current

        # Update Q values
        self.q_values_SA[state, action] += self.alpha * td_error

        self.loss = td_error
        self.logged_loss = False

    def log(self, run):
        if not self.logged_loss:
            run["train/loss"].log(self.loss)
            self.logged_loss = True

    def save(self, dir: str) -> bool:
        with open(f"{dir}/q_values.npy", "wb") as f:
            np.save(f, self.q_values_SA)

    def load(self, dir: str):
        with open(f"{dir}/q_values.npy", "wb") as f:
            self.q_values_SA = np.load(f)
