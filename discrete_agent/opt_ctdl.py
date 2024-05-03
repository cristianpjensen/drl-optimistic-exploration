from agent.utils.scheduler import LinearScheduler
from discrete_agent.discrete_agent import DiscreteAgent
import numpy as np


class OptCTDL(DiscreteAgent):
    def setup(self, config):
        self.n_categories = 51
        self.v_max = 20
        self.v_min = -20
        self.prob_dist_SAN = np.ones((config["n_states"], config["n_actions"], self.n_categories)) / self.n_categories
        self.values_N = np.linspace(self.v_min, self.v_max, self.n_categories)
        self.delta_z = (self.v_max - self.v_min) / (self.n_categories - 1)

        self.alpha = 0.1
        self.gamma = config["gamma"]
        self.scheduler = LinearScheduler([(0, 1), (50_000, 0)])
        self.num_actions = 0

        self.loss = 0
        self.logged_loss = True

    def act(self, state, train):
        # Sample action from categoricals
        if train:
            self.num_actions += 1

            opt_tau = self.scheduler.value(self.num_actions)
            prob_dist_AN = self.prob_dist_SAN[state]
            # Compute CDF
            cdf_AN = np.cumsum(prob_dist_AN, axis=1)
            prob_dist_AN = np.where(cdf_AN >= opt_tau, prob_dist_AN, 0)
            q_values_A = np.sum(self.values_N * prob_dist_AN, axis=1)

            return np.argmax(q_values_A)

        # Compute greedy action
        prob_dist_AN = self.prob_dist_SAN[state]
        q_values_A = np.sum(prob_dist_AN * self.values_N, axis=1)
        return np.argmax(q_values_A)

    def update_policy(self, state, action, reward, next_state, terminal):
        next_action = self.act(next_state, train=False)

        g = reward + (1 - terminal) * self.gamma * self.values_N
        g = np.clip(g, self.v_min + 1e-6, self.v_max - 1e-6)

        # Largest i s.t. values[i] <= g
        i_star = self.values_N.searchsorted(g, side="right") - 1
        zeta = (g - self.values_N[i_star]) / (self.values_N[i_star + 1] - self.values_N[i_star])

        p_hat = np.zeros(self.n_categories)
        # Cannot use simple assignment here, because there might be duplicate indices
        np.add.at(p_hat, i_star, (1 - zeta) * self.prob_dist_SAN[next_state, next_action])
        np.add.at(p_hat, i_star + 1, zeta * self.prob_dist_SAN[next_state, next_action])

        # Bootstrap
        self.prob_dist_SAN[state, action] = (1 - self.alpha) * self.prob_dist_SAN[state, action] + self.alpha * p_hat

    def log(self, run):
        if not self.logged_loss:
            run["train/loss"].log(self.loss)
            self.logged_loss = True

    def save(self, dir: str) -> bool:
        np.save(f"{dir}/prob_dist_SAN.npy", self.prob_dist_SAN)
        return True

    def load(self, dir: str):
        self.prob_dist_SAN = np.load(f"{dir}/prob_dist_SAN.npy")
