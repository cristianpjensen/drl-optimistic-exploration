from agent.utils.scheduler import LinearScheduler
from discrete_agent.discrete_agent import DiscreteAgent
import numpy as np


class OptQTDL(DiscreteAgent):
    def setup(self, config):
        self.n_quantiles = 51
        init_v_min = -20
        init_v_max = 20

        # Initialize values to be uniformly distributed
        self.values_SAN = np.zeros((config["n_states"], config["n_actions"], self.n_quantiles))
        self.values_SAN[:] = np.linspace(init_v_min, init_v_max, self.n_quantiles)
        self.tau_N = (np.arange(1, self.n_quantiles + 1) * 2 - 1) / (2 * self.n_quantiles)

        self.alpha = 0.1
        self.gamma = config["gamma"]
        self.scheduler = LinearScheduler([(0, 1), (50_000, 0)])
        self.num_actions = 0

        self.loss = 0
        self.logged_loss = True

    def act(self, state, train):
        if train:
            self.num_actions += 1

            # Only take the upper quantiles
            values_AN = self.values_SAN[state]
            opt_tau = self.scheduler.value(self.num_actions)
            opt_values = np.where(self.tau_N >= opt_tau, values_AN, 0)
            n_taus = np.sum(self.tau_N >= opt_tau)
            q_values_A = np.sum(opt_values, axis=1) / n_taus

            return np.argmax(q_values_A)

        # Compute greedy action
        q_values_A = np.mean(self.values_SAN[state], axis=1)
        return np.argmax(q_values_A)

    def update_policy(self, state, action, reward, next_state, terminal):
        next_action = self.act(next_state, train=False)

        for i in range(self.n_quantiles):
            value_prime = self.values_SAN[state, action, i]
            tau = self.tau_N[i]

            for j in range(self.n_quantiles):
                g = reward + (1 - terminal) * self.gamma * self.values_SAN[next_state, next_action, j]
                value_prime += (self.alpha / self.n_quantiles) * (tau - (g < self.values_SAN[state, action, i]))

            self.values_SAN[state, action, i] = value_prime

    def log(self, run):
        if not self.logged_loss:
            run["train/loss"].log(self.loss)
            self.logged_loss = True

    def save(self, dir: str) -> bool:
        np.save(f"{dir}/values_SAN.npy", self.values_SAN)
        return True

    def load(self, dir: str):
        self.values_SAN = np.load(f"{dir}/values_SAN.npy")
