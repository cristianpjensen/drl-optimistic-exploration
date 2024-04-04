import torch
from collections import deque
import random
from typing import Tuple


class ReplayBuffer():
    def __init__(self, min_size: int, max_size: int, batch_size: int, device: torch.device = torch.device("cpu")):
        self.buffer = deque(maxlen=max_size)
        self.min_size = min_size
        self.batch_size = batch_size
        self.device = device

    def put(self, transition):
        self.buffer.append(transition)

    def sample_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = random.sample(self.buffer, self.batch_size)
        s_lst, a_lst, r_lst, s_next_lst = [], [], [], []

        for transition in batch:
            s, a, r, s_next = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_next_lst.append(s_next)

        s_batch = torch.tensor(s_lst, dtype=torch.float, device=self.device)
        a_batch = torch.tensor(a_lst, dtype=torch.float, device=self.device)
        r_batch = torch.tensor(r_lst, dtype=torch.float, device=self.device)
        s_prime_batch = torch.tensor(s_next_lst, dtype=torch.float, device=self.device)

        return s_batch, a_batch, r_batch, s_prime_batch

    def size(self):
        return len(self.buffer)

    def is_ready(self):
        return self.size() >= self.min_size
