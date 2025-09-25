import random
import numpy as np
from collections import deque, namedtuple
import torch

class ReplayBuffer:
    def __init__(self, buffer_size=int(1e6), batch_size=256, seed=0, device=None):
        random.seed(seed)
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device or torch.device("cpu")
        self.experience = namedtuple("Experience",
            field_names=["states", "actions", "rewards", "next_states", "dones"])

    def add(self, states, actions, rewards, next_states, dones):
        e = self.experience(states, actions, rewards, next_states, dones)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        def to_tensor(x):
            return torch.from_numpy(np.vstack(x)).float().to(self.device)

        states = to_tensor([e.states for e in experiences])
        actions = to_tensor([e.actions for e in experiences])
        rewards = to_tensor([e.rewards for e in experiences])
        next_states = to_tensor([e.next_states for e in experiences])
        dones = torch.from_numpy(np.vstack([e.dones for e in experiences]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
