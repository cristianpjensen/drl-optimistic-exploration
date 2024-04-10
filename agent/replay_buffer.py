import torch
import numpy as np
from collections import deque, namedtuple
import random
from typing import Tuple


Transition = namedtuple("Transition", ("state", "action", "reward", "next_state"))


class ReplayBuffer():
    """ A simple replay buffer that stores transitions.

    Note: Do not put the stored data on the GPU, since the VRAM would fill up really quick.

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
                torch.tensor(np.array([state]), dtype=torch.float32),
                torch.tensor(action.flatten(), dtype=torch.float32),
                torch.tensor(np.array([reward]), dtype=torch.float32),
                torch.tensor(np.array([next_state]), dtype=torch.float32),
            )
        )

    def sample_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = random.sample(self.buffer, self.batch_size)
        batch = Transition(*zip(*batch))

        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        next_state_batch = torch.cat(batch.next_state).to(self.device)

        return state_batch, action_batch, reward_batch, next_state_batch

    def is_ready(self):
        return self.__len__() >= self.min_size
