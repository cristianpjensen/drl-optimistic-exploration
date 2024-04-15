from typing import Tuple

import gymnasium as gym
import torch


class ReplayBuffer:
    """A simple replay buffer that stores transitions with a frame stacking mechanisms.

    Args:
        observation_space: The observation space of the environment.
        action_space: The action space of the environment.
        min_size: Minimum size of the buffer before we start sampling. Default: 50_000
            (https://www.nature.com/articles/nature14236).
        max_size: Maximum size of the buffer. Default: 1_000_000
            (https://www.nature.com/articles/nature14236).
        batch_size: Number of samples to return when sampling. Default: 32
            (https://www.nature.com/articles/nature14236).
        frame_stack: Number of frames to stack when sampling. Default: 1, set to 4 for Atari games
           (https://www.nature.com/articles/nature14236).
        device: CUDA if available, otherwise CPU.

    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        min_size: int = 50_000,
        max_size: int = 1_000_000,
        batch_size: int = 32,
        frame_stack: int = 1,
        device: torch.device = torch.device("cpu"),
    ):
        self.min_size = min_size
        self.max_size = max_size
        self.batch_size = batch_size
        self.frame_stack = frame_stack
        self.device = device

        # Need to sample observation and action to get the correct pytorch types
        obs_sample = torch.tensor(observation_space.sample())
        act_sample = torch.tensor(action_space.sample())

        self.states = torch.zeros((max_size, *observation_space.shape), dtype=obs_sample.dtype, device=device)
        self.actions = torch.zeros((max_size, *action_space.shape), dtype=act_sample.dtype, device=device)
        self.rewards = torch.zeros((max_size, 1), dtype=torch.float32, device=device)
        self.terminals = torch.zeros((max_size, 1), dtype=torch.bool, device=device)

        # Keep track of which index was last added, since we do not want to sample any frames that
        # contain `last_added`
        self.last_added = 0
        self.entries = 0

    def push(self, state, action, reward, terminal):
        # Implemented as deque
        index = self.entries % self.max_size

        self.states[index] = torch.tensor(state)
        self.actions[index] = torch.tensor(action)
        self.rewards[index] = torch.tensor(reward)
        self.terminals[index] = torch.tensor(terminal)

        self.entries += 1
        self.last_added = index

    def sample_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Use the frame stack by sampling the last frame and then stacking the previous frames

        # Sample random indices
        indices = torch.randint(
            self.frame_stack - 1,
            min(self.entries, self.max_size - 1),
            (self.batch_size,),
        )

        # Do not allow sampling between `last_added+1` (inclusive) and `last_added+frame_stack` (exclusive).
        # So, we do not sample states that contain the last added state, which would overlap between episodes.
        # We do this by sampling between `frame_stack - 1` (to allow stacking) and `size - (frame_stack+1)` (to
        # allow shifting after sampling, such that we avoid the last added frame).
        indices = torch.randint(
            self.frame_stack - 1,
            min(self.entries, min(self.entries, self.max_size - (self.frame_stack+1))),
            (self.batch_size,),
        )
        after_last_added = indices > self.last_added
        indices = torch.where(after_last_added, indices + self.frame_stack, indices)

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
        next_terminal = self.terminals[stacked_and_next_indices]

        # If any frame is terminal, the whole observation is terminal
        next_terminal = next_terminal.reshape(self.batch_size, self.frame_stack + 1).any(dim=1)

        action = self.actions[indices]
        reward = self.rewards[indices]

        # Check for overlapping episodes, which is the case if the current state is terminal
        discard = self.terminals[stacked_indices]
        discard = discard.reshape(self.batch_size, self.frame_stack + 1).any(dim=1)

        keep = discard.logical_not()

        return state[keep], action[keep], reward[keep], next_state[keep], next_terminal[keep]

    def __len__(self):
        return self.entries

    def is_ready(self):
        return len(self) >= self.min_size
