from abc import ABC, abstractmethod
from typing import Tuple
import gymnasium as gym
import numpy as np
import random
import torch
from agent.replay_buffer import ReplayBuffer


class Agent(ABC):
    """ An abstract class that defines the interface for an agent that can interact with an
    environment."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        batch_size: int = 32,
        buffer_size: Tuple[int, int] = (100, 10000),
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.replay_buffer = ReplayBuffer(min_size=buffer_size[0], max_size=buffer_size[1], batch_size=batch_size)

    @abstractmethod
    def setup_agent(self, config):
        """ This method is called when the agent is created. The agent should use the config
        dictionary to set up whatever is needed to learn its policy. """

        pass

    @abstractmethod
    def act(self, state: np.ndarray, train: bool) -> np.int64:
        """ This method is called when the agent needs to select an action to take in the
        environment. """

        pass

    def update_policy(self) -> bool:
        """ This method is called when the agent needs to update its policy. It samples the replay
        buffer and updates the policy. """

        if not self.replay_buffer.is_ready():
            return False

        s_batch, a_batch, r_batch, s_next_batch = self.replay_buffer.sample_batch()
        self.train(s_batch, a_batch, r_batch, s_next_batch)

        return True

    @abstractmethod
    def train(self, s_batch: torch.Tensor, a_batch: torch.Tensor, r_batch: torch.Tensor, s_next_batch: torch.Tensor):
        """ Update policy from a sample batch. """

        pass


class RandomAgent(Agent):
    """ An agent that selects actions uniformly at random. """

    def setup_agent(self, config):
        pass

    def act(self, state, train):
        return self.action_space.sample()

    def train(self, s_batch, a_batch, r_batch, s_next_batch):
        pass
