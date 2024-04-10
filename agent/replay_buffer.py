from typing import Tuple

import torch


class ReplayBuffer:
    """A simple replay buffer that stores transitions with a frame stacking mechanisms.

    Args:
        max_size: Maximum size of the buffer.
        batch_size: Number of samples to return when sampling.
        frame_stack: Number of frames to stack when sampling.
        device: CUDA if available, otherwise CPU.

    """

    def __init__(
        self,
        max_size: int = 1000000,
        batch_size: int = 32,
        frame_stack: int = 1,
        device: torch.device = torch.device("cpu"),
    ):
        self.max_size = max_size
        self.batch_size = batch_size
        self.frame_stack = frame_stack
        self.device = device

        self.states = torch.zeros((max_size, 84, 84), dtype=torch.uint8, device=device)
        self.actions = torch.zeros((max_size, 1), dtype=torch.int64, device=device)
        self.rewards = torch.zeros((max_size, 1), dtype=torch.float32, device=device)
        self.terminals = torch.zeros((max_size, 1), dtype=torch.bool, device=device)

        self.entries = 0

    def push(self, state, action, reward, terminal):
        if self.entries >= self.max_size:
            return

        index = self.entries

        self.states[index] = torch.tensor(state, dtype=torch.uint8, device=self.device)
        self.actions[index] = torch.tensor(action, dtype=torch.int64, device=self.device)
        self.rewards[index] = torch.tensor(reward, dtype=torch.float32, device=self.device)
        self.terminals[index] = torch.tensor(terminal, dtype=torch.bool, device=self.device)

        self.entries += 1

    def sample_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Use the frame stack by sampling the last frame and then stacking the previous frames

        # Sample random indices
        indices = torch.randint(self.frame_stack - 1, min(self.entries, self.max_size), (self.batch_size,))

        # Stack frames
        stacked_indices = (indices.reshape(-1, 1) + torch.arange(-(self.frame_stack - 1), 1)).flatten()
        state = self.states[stacked_indices]
        state = state.reshape(self.batch_size, self.frame_stack, *state.shape[1:])

        # Next state with stacked frames aswell but shifted one
        next_state = self.states[stacked_indices + 1]
        next_state = next_state.reshape(self.batch_size, self.frame_stack, *next_state.shape[1:])

        # Mark as terminal if any of the frames in the stack or next state is terminal. These should
        # not be used for learning.
        stacked_and_next_indices = (indices.reshape(-1, 1) + torch.arange(-(self.frame_stack - 1), 2)).flatten()
        terminal = self.terminals[stacked_and_next_indices]
        terminal = terminal.reshape(self.batch_size, self.frame_stack + 1).any(dim=1)

        action = self.actions[indices]
        reward = self.rewards[indices]

        return state, action, reward, next_state, terminal

    def __len__(self):
        return self.entries

    def is_ready(self):
        return len(self) >= self.batch_size + self.frame_stack
