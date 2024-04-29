from typing import List, Tuple

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
        device: CUDA if available, otherwise CPU.

    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        min_size: int = 50_000,
        max_size: int = 1_000_000,
        batch_size: int = 32,
        device: torch.device = torch.device("cpu"),
    ):
        self.min_size = min_size
        self.max_size = max_size
        self.batch_size = batch_size
        self.device = device

        # Need to sample observation and action to get the correct pytorch types
        obs_type = torch.tensor(observation_space.sample()).dtype
        act_type = torch.tensor(action_space.sample()).dtype

        # Assume that the observation space is a stack of frames
        self.frame_stack = observation_space.shape[0]
        frame_shape = observation_space.shape[1:]

        self.frames = torch.zeros((max_size, *frame_shape), dtype=obs_type, device=device)
        self.actions = torch.zeros((max_size, *action_space.shape), dtype=act_type, device=device)
        self.rewards = torch.zeros((max_size,), dtype=torch.float32, device=device)
        self.terminals = torch.zeros((max_size,), dtype=torch.bool, device=device)

        # Either 1 or 0, depending on whether the index would result in a valid frame stack that has
        # no overlapping
        self.sample_prob = torch.zeros((max_size,), dtype=torch.uint8, device=device)

        # Keep track of how many frames have been added
        self.entries = 0
        self.added = 0

    def push_episode(self, episode: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
        """Pushes an entire episode to the replay buffer."""
        for state, action, reward, terminal in episode:
            self.push(state, action, reward, terminal)

    def push(self, state, action, reward, terminal):
        """Pushes to the replay buffer, in a manner such that we do not store duplicate frames and
        have no frame stacks that contain frames from more than one episode."""

        self.added += 1

        index = self.entries % self.max_size

        # If this is the first index, we can put in the full history, since otherwise we won't have
        # the history for the current transition.
        if index == 0:
            self.entries = self.frame_stack - 1
            self._push_frame(state, action, reward, terminal, full_observation=True)
            return
        
        # If previous transition was terminal, it means we have started a new episode, which means
        # that we need to skip `frame_stack` frames, since we do not want transitions to overlap
        # between episodes. And, we need to put in the full observation, so we have the history of
        # the current state.
        if self.terminals[index-1]:
            self.entries += self.frame_stack - 1

            new_index = self.entries % self.max_size

            # Make sure there is space to put the full observation
            if new_index < self.frame_stack - 1:
                self.entries = self.frame_stack - 1

            self._push_frame(state, action, reward, terminal, full_observation=True)
            return

        # Just add the frame, since we already have the history
        self._push_frame(state, action, reward, terminal)

    def _push_frame(self, state, action, reward, terminal, full_observation=False):
        index = self.entries % self.max_size

        # The states are frame stacked, so we need to push the only the last frame, since the
        # previous frames have already been pushed in previous iterations
        if full_observation:
            self.frames[index-self.frame_stack+1:index+1] = torch.tensor(state)
        else:
            self.frames[index] = torch.tensor(state[-1])

        self.actions[index] = torch.tensor(action)
        # Bin rewards to { -1, 0, 1 } by its sign
        self.rewards[index] = torch.sign(torch.tensor(reward))
        self.terminals[index] = torch.tensor(terminal)

        # Make sure that the next `frame_stack` frames are not sampled, because that would result in
        # overlapping stacks from different episodes
        prob_insertion = torch.zeros(min(self.max_size - index, self.frame_stack), dtype=torch.uint8)
        prob_insertion[0] = 1
        self.sample_prob[index:min(index + self.frame_stack, self.max_size)] = prob_insertion

        self.entries += 1

    def sample_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (state, action, reward, next_state, terminal) batches."""

        # Use the frame stack by sampling the last frame and then stacking the previous frames.
        # Also, make sure not to sample last index, since then we won't know the next state.
        indices = torch.multinomial(self.sample_prob[:-2].float(), self.batch_size)
        stacked_indices = indices.reshape(-1, 1) + torch.arange(-(self.frame_stack - 1), 1).to(self.device)

        return self.frames[stacked_indices], self.actions[indices], self.rewards[indices], self.frames[stacked_indices+1], self.terminals[indices]

    def __len__(self):
        return min(self.added, self.max_size)

    def is_ready(self):
        return len(self) >= self.min_size
