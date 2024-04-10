from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np
import torch

from agent.replay_buffer import ReplayBuffer


class Agent(ABC):
    """An abstract class that defines the interface for an agent that can interact with an
    environment."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        batch_size: int = 32,
        buffer_size: int = 1000000,
        frame_stack: int = 1,
        device: torch.device = torch.device("cpu"),
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.replay_buffer = ReplayBuffer(
            max_size=buffer_size,
            frame_stack=frame_stack,
            batch_size=batch_size,
            device=device,
        )
        self.device = device

    @abstractmethod
    def setup(self, config: dict):
        """This method is called when the agent is created. The agent should use the config
        dictionary to set up whatever is needed to learn its policy."""

        pass

    @abstractmethod
    def act(self, state: np.ndarray, train: bool) -> np.int64:
        """This method is called when the agent needs to select an action to take in the
        environment."""

        pass

    def update_policy(self) -> bool:
        """This method is called when the agent needs to update its policy. It samples the replay
        buffer and updates the policy."""

        if not self.replay_buffer.is_ready():
            return False

        s_batch, a_batch, r_batch, s_next_batch, terminal_batch = self.replay_buffer.sample_batch()
        self.train(s_batch, a_batch, r_batch, s_next_batch, terminal_batch)

        return True

    @abstractmethod
    def train(
        self,
        s_batch: torch.Tensor,
        a_batch: torch.Tensor,
        r_batch: torch.Tensor,
        s_next_batch: torch.Tensor,
        terminal_batch: torch.Tensor,
    ):
        """Update policy from a sample batch."""

        pass

    @abstractmethod
    def log(self, state: np.ndarray, action: np.int64):
        """Log anything relevant to the agent."""

        pass


class RandomAgent(Agent):
    """An agent that selects actions uniformly at random."""

    def setup(self, config):
        pass

    def act(self, state, timestep, train):
        return self.action_space.sample()

    def train(self, s_batch, a_batch, r_batch, s_next_batch, terminal_batch):
        pass

    def log(self, state, action):
        pass
