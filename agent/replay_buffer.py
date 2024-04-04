import torch
import numpy as np
from collections import deque, namedtuple
import random
from typing import Tuple


Transition = namedtuple("Transition", ("state", "action", "reward", "next_state"))


class ReplayBuffer():
    """ A simple replay buffer that stores transitions.

    Args:
        min_size: Minimum size of the buffer before sampling.
        max_size: Maximum size of the buffer.
        batch_size: Number of samples to return when sampling. Make sure it is smaller than
            `min_size`.
        device: CUDA if available, otherwise CPU.

    """

    def __init__(self, min_size: int, max_size: int, batch_size: int, device: torch.device = torch.device("cpu")):
        self.buffer = deque(maxlen=max_size)
        self.min_size = min_size
        self.batch_size = batch_size
        self.device = device

    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, reward, next_state):
        self.buffer.append(
            Transition(
                torch.tensor(state, device=self.device),
                torch.tensor(action.flatten(), device=self.device),
                torch.tensor([reward], device=self.device),
                torch.tensor(next_state, device=self.device),
            )
        )

    def sample_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = random.sample(self.buffer, self.batch_size)
        batch = Transition(*zip(*batch))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)

        return state_batch, action_batch, reward_batch, next_state_batch

    def is_ready(self):
        return self.__len__() >= self.min_size
