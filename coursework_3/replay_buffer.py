# replay_buffer.py
import random
import numpy as np
from collections import deque, namedtuple
import torch


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples for multi-agent MADDPG."""

    def __init__(self, buffer_size=int(1e6), batch_size=256, seed=0, device=None):
        """
        Args
        ----
        buffer_size : int
            Maximum number of experiences to store.
        batch_size : int
            Size of each training batch.
        seed : int
            Random seed for sampling reproducibility.
        device : torch.device
            Device to move sampled tensors to.
        """
        random.seed(seed)
        np.random.seed(seed)
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device or torch.device("cpu")

        # Store full joint views (concat states/actions) + per-agent reward/done vectors
        self.experience = namedtuple(
            "Experience",
            field_names=["states", "actions", "rewards", "next_states", "dones"],
        )

    def add(self, states, actions, rewards, next_states, dones):
        """
        Add a new experience to memory.

        Parameters
        ----------
        states      : np.ndarray shape (S_all,)
        actions     : np.ndarray shape (A_all,)
        rewards     : np.ndarray shape (n_agents,)   # per-agent
        next_states : np.ndarray shape (S_all,)
        dones       : np.ndarray shape (n_agents,)   # per-agent (0/1)
        """
        e = self.experience(states, actions, rewards, next_states, dones)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences and return tensors on self.device."""
        experiences = random.sample(self.memory, k=self.batch_size)

        def vstack_float(xs):
            return torch.from_numpy(np.vstack(xs)).float().to(self.device)

        # (B, S_all)
        states = vstack_float([e.states for e in experiences])

        # (B, A_all)
        actions = vstack_float([e.actions for e in experiences])

        # (B, n_agents)  ✅ per-agent rewards preserved
        rewards = vstack_float([e.rewards for e in experiences])

        # (B, S_all)
        next_states = vstack_float([e.next_states for e in experiences])

        # (B, n_agents)  ✅ per-agent dones preserved (as float 0/1)
        dones_np = np.vstack([e.dones for e in experiences]).astype(np.uint8)
        dones = torch.from_numpy(dones_np).float().to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


# import random
# import numpy as np
# from collections import deque, namedtuple
# import torch

# class ReplayBuffer:
#     def __init__(self, buffer_size=int(1e6), batch_size=256, seed=0, device=None):
#         random.seed(seed)
#         self.memory = deque(maxlen=buffer_size)
#         self.batch_size = batch_size
#         self.device = device or torch.device("cpu")
#         self.experience = namedtuple("Experience",
#             field_names=["states", "actions", "rewards", "next_states", "dones"])

#     def add(self, states, actions, rewards, next_states, dones):
#         e = self.experience(states, actions, rewards, next_states, dones)
#         self.memory.append(e)

#     # def sample(self):
#     #     experiences = random.sample(self.memory, k=self.batch_size)

#     #     def to_tensor(x):
#     #         return torch.from_numpy(np.vstack(x)).float().to(self.device)

#     #     states = to_tensor([e.states for e in experiences])
#     #     actions = to_tensor([e.actions for e in experiences])
#     #     rewards = to_tensor([e.rewards for e in experiences])
#     #     next_states = to_tensor([e.next_states for e in experiences])
#     #     dones = torch.from_numpy(np.vstack([e.dones for e in experiences]).astype(np.uint8)).float().to(self.device)

#     #     return (states, actions, rewards, next_states, dones)


#     def sample(self):
#         experiences = random.sample(self.memory, k=self.batch_size)

#         def to_tensor(x):  # stacks along first dim
#             return torch.from_numpy(np.vstack(x)).float().to(self.device)

#         states      = to_tensor([e.states       for e in experiences])      # (B, S_all)
#         actions     = to_tensor([e.actions      for e in experiences])      # (B, A_all)
#         rewards     = to_tensor([e.rewards      for e in experiences])      # (B, n_agents)  ✅
#         next_states = to_tensor([e.next_states  for e in experiences])      # (B, S_all)
#         dones = torch.from_numpy(
#             np.vstack([e.dones for e in experiences]).astype(np.uint8)
#         ).float().to(self.device)                                           # (B, n_agents)  ✅

#         return (states, actions, rewards, next_states, dones)

#     def __len__(self):
#         return len(self.memory)
