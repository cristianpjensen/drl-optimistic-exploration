from agent.utils.scheduler import LinearScheduler
from discrete_agent.discrete_agent import DiscreteAgent
import numpy as np


class QTDL(DiscreteAgent):
    def setup(self, config):
        self.n_quantiles = 32

        # Initialize values as Dirac at 0
        self.values_SAN = np.zeros((config["n_states"], config["n_actions"], self.n_quantiles))
        self.tau_N = np.arange(self.n_quantiles) / self.n_quantiles

        self.alpha = 0.1
        self.gamma = config["gamma"]
        self.scheduler = LinearScheduler([(0, 1), (50_000, 0)])
        self.num_actions = 0

        self.loss = 0
        self.logged_loss = True

    def act(self, state, train):
        if train:
            self.num_actions += 1

        if train and np.random.random() < self.scheduler.value(self.num_actions):
            return self.action_space.sample()

        # Compute greedy action
        q_values_A = np.mean(self.values_SAN[state], axis=1)
        return np.argmax(q_values_A)

    def update_policy(self, state, action, reward, next_state, terminal):
        next_action = self.act(next_state, train=False)

        # Ref: https://arxiv.org/pdf/1710.10044 (Equation 12)

        indices = np.arange(self.n_quantiles)
        tau_hat_N = np.where(indices == 0, 0, (self.tau_N[indices - 1] + self.tau_N[indices]) / 2)
        next_values_N = self.values_SAN[next_state, next_action]
        target_N = reward + (1 - terminal) * self.gamma * next_values_N
        
        quantile_loss_NN = tau_hat_N.reshape(-1, 1) - (target_N.reshape(1, -1) < self.values_SAN[state, action].reshape(-1, 1))
        update_N = np.mean(quantile_loss_NN, axis=1)
        self.values_SAN[state, action] += self.alpha * update_N

    def log(self, run):
        if not self.logged_loss:
            run["train/loss"].log(self.loss)
            self.logged_loss = True

    def save(self, dir: str) -> bool:
        np.save(f"{dir}/values_SAN.npy", self.values_SAN)
        return True

    def load(self, dir: str):
        self.values_SAN = np.load(f"{dir}/values_SAN.npy")
