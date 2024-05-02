from agent.utils.scheduler import LinearScheduler
from discrete_agent.discrete_agent import DiscreteAgent
import numpy as np


class QTDL(DiscreteAgent):
    def setup(self, config):
        self.n_quantiles = 51
        self.thetas_SAQ = np.zeros((config["n_states"], config["n_actions"], self.n_quantiles))

        self.tau_Q = (np.arange(self.n_quantiles) * 2 + 1) / (2 * self.n_quantiles)
        self.kappa = 1

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
        q_values_A = np.mean(self.thetas_SAQ[state], axis=1)
        return np.argmax(q_values_A)

    def update_policy(self, state, action, reward, next_state, terminal):
        q_values_A = np.mean(self.thetas_SAQ[next_state], axis=1)
        next_action = np.argmax(q_values_A) 
        target = reward + (1 - terminal) * self.gamma * self.thetas_SAQ[next_state, next_action]

        # [target, current]
        error_QQ = self.tau_Q.reshape(1, -1) - (target.reshape(-1, 1) < self.thetas_SAQ[state, action].reshape(1, -1))
        error_Q = np.mean(error_QQ, axis=1)
        self.thetas_SAQ[state, action] += self.alpha * error_Q

        self.loss = np.mean(error_Q)
        self.logged_loss = False

    def log(self, run):
        if not self.logged_loss:
            run["train/loss"].log(self.loss)
            self.logged_loss = True

    def save(self, dir: str) -> bool:
        np.save(f"{dir}/thetas_SAQ.npy", self.thetas_SAQ)

    def load(self, dir: str):
        self.thetas_SAQ = np.load(f"{dir}/thetas_SAQ.npy")
