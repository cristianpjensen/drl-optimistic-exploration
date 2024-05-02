from agent.utils.scheduler import LinearScheduler
from discrete_agent.discrete_agent import DiscreteAgent
import numpy as np


class OptCTDL(DiscreteAgent):
    def setup(self, config):
        self.n_categories = 51
        self.v_max = 1
        self.v_min = -20
        self.prob_dist_SAN = np.zeros((config["n_states"], config["n_actions"], self.n_categories))
        # Initialize all state, action pairs optimistically
        self.prob_dist_SAN[:, :, -1] = 1
        self.values_N = np.linspace(self.v_min, self.v_max, self.n_categories)
        self.delta_z = (self.v_max - self.v_min) / (self.n_categories - 1)

        self.alpha = 0.1
        self.gamma = config["gamma"]
        self.scheduler = LinearScheduler([(0, 1), (50_000, 0)])
        self.num_actions = 0

        self.loss = 0
        self.logged_loss = True

    def act(self, state, train):
        if train:
            self.num_actions += 1

        # Sample action from categoricals
        if train and np.random.random() < self.scheduler.value(self.num_actions):
            prob_dist_AN = self.prob_dist_SAN[state]
            samples_A = np.zeros(prob_dist_AN.shape[0])
            for i in range(prob_dist_AN.shape[0]):
                samples_A[i] = np.random.choice(self.values_N, p=prob_dist_AN[i])

            return np.argmax(samples_A)

        # Compute greedy action
        prob_dist_AN = self.prob_dist_SAN[state]
        q_values_A = np.sum(prob_dist_AN * self.values_N, axis=1)
        return np.argmax(q_values_A)

    def update_policy(self, state, action, reward, next_state, terminal):
        next_action = self.act(next_state, train=False)

        p_hat = np.zeros(self.n_categories)
        for j in range(self.n_categories):
            g = reward + (1 - terminal) * self.gamma * self.values_N[j]

            if g <= self.v_min:
                p_hat[0] += self.prob_dist_SAN[next_state, next_action, j]
            elif g >= self.v_max:
                p_hat[-1] += self.prob_dist_SAN[next_state, next_action, j]
            else:
                # Largest i such that values[i] <= g
                i_star = int((g - self.v_min) / self.delta_z)
                zeta = (g - self.values_N[i_star]) / self.delta_z

                p_hat[i_star] += (1 - zeta) * self.prob_dist_SAN[next_state, next_action, j]
                p_hat[i_star + 1] += zeta * self.prob_dist_SAN[next_state, next_action, j]

        for i in range(self.n_categories):
            self.prob_dist_SAN[state, action, i] = (1 - self.alpha) * self.prob_dist_SAN[state, action, i] + self.alpha * p_hat[i]

    def log(self, run):
        if not self.logged_loss:
            run["train/loss"].log(self.loss)
            self.logged_loss = True

    def save(self, dir: str) -> bool:
        np.save(f"{dir}/prob_dist_SAN.npy", self.prob_dist_SAN)

    def load(self, dir: str):
        self.prob_dist_SAN = np.load(f"{dir}/prob_dist_SAN.npy")
