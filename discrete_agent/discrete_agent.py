from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np
from neptune import Run

class DiscreteAgent(ABC):
    def __init__(self, observation_space: gym.Space, action_space: gym.Space):
        self.observation_space = observation_space
        self.action_space = action_space
        self.training_error = []

    @abstractmethod
    def setup(self, config: dict):
        pass

    @abstractmethod
    def act(self, state: np.int64, train: bool) -> np.int64:
        pass

    @abstractmethod
    def update_policy(self, state: np.int64, action: np.int64, reward: np.float64, next_state: np.int64, terminal: bool):
        pass

    @abstractmethod
    def log(self, run: Run):
        pass

    @abstractmethod
    def save(self, dir: str) -> bool:
        pass

    @abstractmethod
    def load(self, dir: str):
        pass
