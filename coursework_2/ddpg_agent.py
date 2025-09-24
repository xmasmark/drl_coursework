# coursework_2/ddpg_agent.py
import random
import numpy as np
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.optim as optim

from model import Actor, Critic

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ------------------------------ Running State Normalization ------------------------------
class RunningNorm:
    def __init__(self, size, eps=1e-5):
        self.mean = np.zeros(size, dtype=np.float64)
        self.var = np.ones(size, dtype=np.float64)
        self.count = eps

    def update(self, x: np.ndarray):
        x = np.atleast_2d(x).astype(np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * (batch_count / tot_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta**2) * (self.count * batch_count / tot_count)
        new_var = M2 / tot_count

        self.mean, self.var, self.count = new_mean, new_var, tot_count

    def normalize(self, x: np.ndarray):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


# -------------------------------------- Replay Buffer -------------------------------------
class ReplayBuffer:
    def __init__(self, action_size, buffer_size=int(1e6), batch_size=256, seed=0):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )
        random.seed(seed)
        np.random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        exps = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(
            np.vstack([e.state for e in exps]).astype(np.float32)
        ).to(DEVICE)
        actions = torch.from_numpy(
            np.vstack([e.action for e in exps]).astype(np.float32)
        ).to(DEVICE)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in exps]).astype(np.float32)
        ).to(DEVICE)
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in exps]).astype(np.float32)
        ).to(DEVICE)
        dones = torch.from_numpy(
            np.vstack([e.done for e in exps]).astype(np.uint8)
        ).float().to(DEVICE)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


# ---------------------------------- Ornstein–Uhlenbeck Noise ------------------------------
class OUNoise:
    def __init__(self, size, seed=0, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size, dtype=np.float32)
        self.theta = theta
        self.sigma = sigma
        self.state = self.mu.copy()
        random.seed(seed)
        np.random.seed(seed)

    def reset(self):
        self.state = self.mu.copy()

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(len(self.state))
        self.state += dx
        return self.state


# ------------------------------------------- Agent ----------------------------------------
class Agent:
    """
    DDPG Agent (single or 20-agent loop).
    """

    def __init__(
        self,
        state_size,
        action_size,
        random_seed=2,
        # Tuned defaults
        actor_lr=1e-4,
        critic_lr=1e-3,
        weight_decay=0.0,        # <— set to 0.0 for critic Adam (was 1e-2)
        gamma=0.99,
        tau=1e-3,
        batch_size=256,
        buffer_size=int(1e6),
        start_learn_after=500,   # earlier warmup (train_20 can override)
        updates_per_step=1,
        noise_sigma=0.15,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.updates_per_step = updates_per_step
        self.start_learn_after = start_learn_after

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

        # Running state normalization
        self.state_norm = RunningNorm(state_size)

        # Actor
        self.actor_local = Actor(state_size, action_size, seed=random_seed).to(DEVICE)
        self.actor_target = Actor(state_size, action_size, seed=random_seed).to(DEVICE)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=actor_lr)

        # Critic
        self.critic_local = Critic(state_size, action_size, seed=random_seed).to(DEVICE)
        self.critic_target = Critic(state_size, action_size, seed=random_seed).to(DEVICE)
        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(), lr=critic_lr, weight_decay=weight_decay
        )

        # Initialize target = local
        self.hard_update(self.actor_target, self.actor_local)
        self.hard_update(self.critic_target, self.critic_local)

        # Replay buffer & noise
        self.memory = ReplayBuffer(
            action_size, buffer_size=buffer_size, batch_size=batch_size, seed=random_seed
        )
        self.noise = OUNoise(action_size, seed=random_seed, sigma=noise_sigma)
        self.noise_scale = 1.2  # start a bit higher for exploration

    # ------------------------------- Utils -------------------------------
    def hard_update(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(s.data)

    @staticmethod
    def soft_update(local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    # ------------------------------- Agent API ---------------------------
    def step(self, state, action, reward, next_state, done, t):
        """
        Single-agent training path: add one transition and maybe learn.
        (In 20-agent training we add transitions externally and call learn once per env step.)
        """
        self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) > self.start_learn_after:
            for _ in range(self.updates_per_step):
                experiences = self.memory.sample()
                self.learn(experiences)

    def act(self, state, add_noise=True):
        """
        Returns actions for given state as per current policy.
        Applies running normalization to the state.
        """
        # Update running stats with streaming state & normalize for inference
        self.state_norm.update(state)
        nstate = self.state_norm.normalize(state)

        state_t = torch.from_numpy(nstate).float().unsqueeze(0).to(DEVICE)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state_t).cpu().numpy().squeeze(0)
        self.actor_local.train()

        if add_noise:
            action = action + self.noise_scale * self.noise.sample()
        return np.clip(action, -1.0, 1.0)

    def reset(self):
        self.noise.reset()
        # gentle noise decay, keep a higher exploration floor early on
        self.noise_scale = max(0.10, self.noise_scale * 0.999)

    def learn(self, experiences):
        """
        Update actor and critic using sampled batch of experiences.
        Applies normalization to states (but does NOT update running stats here).
        Includes TD3-lite target policy smoothing for stability.
        """
        states, actions, rewards, next_states, dones = experiences

        # Normalize states via running stats (no stat updates here)
        with torch.no_grad():
            s_cpu = states.detach().cpu().numpy()
            ns_cpu = next_states.detach().cpu().numpy()
            s_n = torch.from_numpy(self.state_norm.normalize(s_cpu)).float().to(DEVICE)
            ns_n = torch.from_numpy(self.state_norm.normalize(ns_cpu)).float().to(DEVICE)

        # ----- Critic update -----
        with torch.no_grad():
            # TD3-lite: target policy smoothing
            next_actions = self.actor_target(ns_n)
            noise = (0.2 * torch.randn_like(next_actions)).clamp(-0.5, 0.5)
            next_actions = (next_actions + noise).clamp(-1.0, 1.0)

            Q_targets_next = self.critic_target(ns_n, next_actions)
            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        Q_expected = self.critic_local(s_n, actions)
        critic_loss = nn.SmoothL1Loss()(Q_expected, Q_targets)  # Huber loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1.0)
        self.critic_optimizer.step()

        # ----- Actor update -----
        actions_pred = self.actor_local(s_n)
        actor_loss = -self.critic_local(s_n, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1.0)  # clip actor too
        self.actor_optimizer.step()

        # ----- Soft update -----
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)


# # coursework_2/ddpg_agent.py
# import random
# import numpy as np
# from collections import deque, namedtuple

# import torch
# import torch.nn as nn
# import torch.optim as optim

# from model import Actor, Critic

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# # ------------------------------ Running State Normalization ------------------------------
# class RunningNorm:
#     def __init__(self, size, eps=1e-5):
#         self.mean = np.zeros(size, dtype=np.float64)
#         self.var = np.ones(size, dtype=np.float64)
#         self.count = eps

#     def update(self, x: np.ndarray):
#         x = np.atleast_2d(x).astype(np.float64)
#         batch_mean = x.mean(axis=0)
#         batch_var = x.var(axis=0)
#         batch_count = x.shape[0]

#         delta = batch_mean - self.mean
#         tot_count = self.count + batch_count

#         new_mean = self.mean + delta * (batch_count / tot_count)
#         m_a = self.var * self.count
#         m_b = batch_var * batch_count
#         M2 = m_a + m_b + (delta**2) * (self.count * batch_count / tot_count)
#         new_var = M2 / tot_count

#         self.mean, self.var, self.count = new_mean, new_var, tot_count

#     def normalize(self, x: np.ndarray):
#         return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


# # -------------------------------------- Replay Buffer -------------------------------------
# class ReplayBuffer:
#     def __init__(self, action_size, buffer_size=int(1e6), batch_size=256, seed=0):
#         self.action_size = action_size
#         self.memory = deque(maxlen=buffer_size)
#         self.batch_size = batch_size
#         self.experience = namedtuple(
#             "Experience",
#             field_names=["state", "action", "reward", "next_state", "done"],
#         )
#         random.seed(seed)
#         np.random.seed(seed)

#     def add(self, state, action, reward, next_state, done):
#         e = self.experience(state, action, reward, next_state, done)
#         self.memory.append(e)

#     def sample(self):
#         exps = random.sample(self.memory, k=self.batch_size)
#         states = torch.from_numpy(
#             np.vstack([e.state for e in exps]).astype(np.float32)
#         ).to(DEVICE)
#         actions = torch.from_numpy(
#             np.vstack([e.action for e in exps]).astype(np.float32)
#         ).to(DEVICE)
#         rewards = torch.from_numpy(
#             np.vstack([e.reward for e in exps]).astype(np.float32)
#         ).to(DEVICE)
#         next_states = torch.from_numpy(
#             np.vstack([e.next_state for e in exps]).astype(np.float32)
#         ).to(DEVICE)
#         dones = torch.from_numpy(
#             np.vstack([e.done for e in exps]).astype(np.uint8)
#         ).float().to(DEVICE)
#         return (states, actions, rewards, next_states, dones)

#     def __len__(self):
#         return len(self.memory)


# # ---------------------------------- Ornstein–Uhlenbeck Noise ------------------------------
# class OUNoise:
#     def __init__(self, size, seed=0, mu=0.0, theta=0.15, sigma=0.2):
#         self.mu = mu * np.ones(size, dtype=np.float32)
#         self.theta = theta
#         self.sigma = sigma
#         self.state = self.mu.copy()
#         random.seed(seed)
#         np.random.seed(seed)

#     def reset(self):
#         self.state = self.mu.copy()

#     def sample(self):
#         dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(len(self.state))
#         self.state += dx
#         return self.state


# # ------------------------------------------- Agent ----------------------------------------
# class Agent:
#     """
#     DDPG Agent (single or 20-agent loop).
#     """

#     def __init__(
#         self,
#         state_size,
#         action_size,
#         random_seed=2,
#         # Tuned defaults
#         actor_lr=1e-4,
#         critic_lr=1e-3,
#         weight_decay=1e-2,
#         gamma=0.99,
#         tau=1e-3,
#         batch_size=256,
#         buffer_size=int(1e6),
#         start_learn_after=500,   # earlier warmup (train_20 can override)
#         updates_per_step=1,
#         noise_sigma=0.15,
#     ):
#         self.state_size = state_size
#         self.action_size = action_size
#         self.gamma = gamma
#         self.tau = tau
#         self.batch_size = batch_size
#         self.updates_per_step = updates_per_step
#         self.start_learn_after = start_learn_after

#         torch.manual_seed(random_seed)
#         np.random.seed(random_seed)
#         random.seed(random_seed)

#         # Running state normalization
#         self.state_norm = RunningNorm(state_size)

#         # Actor
#         self.actor_local = Actor(state_size, action_size, seed=random_seed).to(DEVICE)
#         self.actor_target = Actor(state_size, action_size, seed=random_seed).to(DEVICE)
#         self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=actor_lr)

#         # Critic
#         self.critic_local = Critic(state_size, action_size, seed=random_seed).to(DEVICE)
#         self.critic_target = Critic(state_size, action_size, seed=random_seed).to(DEVICE)
#         self.critic_optimizer = optim.Adam(
#             self.critic_local.parameters(), lr=critic_lr, weight_decay=weight_decay
#         )

#         # Initialize target = local
#         self.hard_update(self.actor_target, self.actor_local)
#         self.hard_update(self.critic_target, self.critic_local)

#         # Replay buffer & noise
#         self.memory = ReplayBuffer(
#             action_size, buffer_size=buffer_size, batch_size=batch_size, seed=random_seed
#         )
#         self.noise = OUNoise(action_size, seed=random_seed, sigma=noise_sigma)
#         self.noise_scale = 1.0  # gentle decay over episodes

#     # ------------------------------- Utils -------------------------------
#     def hard_update(self, target, source):
#         for t, s in zip(target.parameters(), source.parameters()):
#             t.data.copy_(s.data)

#     @staticmethod
#     def soft_update(local_model, target_model, tau):
#         for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
#             target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

#     # ------------------------------- Agent API ---------------------------
#     def step(self, state, action, reward, next_state, done, t):
#         """
#         Single-agent training path: add one transition and maybe learn.
#         (In 20-agent training we add transitions externally and call learn once per env step.)
#         """
#         self.memory.add(state, action, reward, next_state, done)
#         if len(self.memory) > self.start_learn_after:
#             for _ in range(self.updates_per_step):
#                 experiences = self.memory.sample()
#                 self.learn(experiences)

#     def act(self, state, add_noise=True):
#         """
#         Returns actions for given state as per current policy.
#         Applies running normalization to the state.
#         """
#         # Update running stats with streaming state & normalize for inference
#         self.state_norm.update(state)
#         nstate = self.state_norm.normalize(state)

#         state_t = torch.from_numpy(nstate).float().unsqueeze(0).to(DEVICE)
#         self.actor_local.eval()
#         with torch.no_grad():
#             action = self.actor_local(state_t).cpu().numpy().squeeze(0)
#         self.actor_local.train()

#         if add_noise:
#             action = action + self.noise_scale * self.noise.sample()
#         return np.clip(action, -1.0, 1.0)

#     def reset(self):
#         self.noise.reset()
#         # gentle noise decay
#         self.noise_scale = max(0.05, self.noise_scale * 0.999)

#     def learn(self, experiences):
#         """
#         Update actor and critic using sampled batch of experiences.
#         Applies normalization to states (but does NOT update running stats here).
#         Includes TD3-lite target policy smoothing for stability.
#         """
#         states, actions, rewards, next_states, dones = experiences

#         # Normalize states via running stats (no stat updates here)
#         with torch.no_grad():
#             s_cpu = states.detach().cpu().numpy()
#             ns_cpu = next_states.detach().cpu().numpy()
#             s_n = torch.from_numpy(self.state_norm.normalize(s_cpu)).float().to(DEVICE)
#             ns_n = torch.from_numpy(self.state_norm.normalize(ns_cpu)).float().to(DEVICE)

#         # ----- Critic update -----
#         with torch.no_grad():
#             # TD3-lite: target policy smoothing
#             next_actions = self.actor_target(ns_n)
#             noise = (0.2 * torch.randn_like(next_actions)).clamp(-0.5, 0.5)
#             next_actions = (next_actions + noise).clamp(-1.0, 1.0)

#             Q_targets_next = self.critic_target(ns_n, next_actions)
#             Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

#         Q_expected = self.critic_local(s_n, actions)
#         critic_loss = nn.MSELoss()(Q_expected, Q_targets)
#         self.critic_optimizer.zero_grad()
#         critic_loss.backward()
#         nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1.0)
#         self.critic_optimizer.step()

#         # ----- Actor update -----
#         actions_pred = self.actor_local(s_n)
#         actor_loss = -self.critic_local(s_n, actions_pred).mean()
#         self.actor_optimizer.zero_grad()
#         actor_loss.backward()
#         nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1.0)  # clip actor too
#         self.actor_optimizer.step()

#         # ----- Soft update -----
#         self.soft_update(self.critic_local, self.critic_target, self.tau)
#         self.soft_update(self.actor_local, self.actor_target, self.tau)



# # # coursework_2/ddpg_agent.py
# # import random
# # import numpy as np
# # from collections import deque, namedtuple

# # import torch
# # import torch.nn as nn
# # import torch.optim as optim

# # from model import Actor, Critic

# # DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# # # ------------------------------ Running State Normalization ------------------------------
# # class RunningNorm:
# #     def __init__(self, size, eps=1e-5):
# #         self.mean = np.zeros(size, dtype=np.float64)
# #         self.var = np.ones(size, dtype=np.float64)
# #         self.count = eps

# #     def update(self, x: np.ndarray):
# #         x = np.atleast_2d(x).astype(np.float64)
# #         batch_mean = x.mean(axis=0)
# #         batch_var = x.var(axis=0)
# #         batch_count = x.shape[0]

# #         delta = batch_mean - self.mean
# #         tot_count = self.count + batch_count

# #         new_mean = self.mean + delta * (batch_count / tot_count)
# #         m_a = self.var * self.count
# #         m_b = batch_var * batch_count
# #         M2 = m_a + m_b + (delta**2) * (self.count * batch_count / tot_count)
# #         new_var = M2 / tot_count

# #         self.mean, self.var, self.count = new_mean, new_var, tot_count

# #     def normalize(self, x: np.ndarray):
# #         return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


# # # -------------------------------------- Replay Buffer -------------------------------------
# # class ReplayBuffer:
# #     def __init__(self, action_size, buffer_size=int(1e6), batch_size=256, seed=0):
# #         self.action_size = action_size
# #         self.memory = deque(maxlen=buffer_size)
# #         self.batch_size = batch_size
# #         self.experience = namedtuple(
# #             "Experience",
# #             field_names=["state", "action", "reward", "next_state", "done"],
# #         )
# #         random.seed(seed)
# #         np.random.seed(seed)

# #     def add(self, state, action, reward, next_state, done):
# #         e = self.experience(state, action, reward, next_state, done)
# #         self.memory.append(e)

# #     def sample(self):
# #         exps = random.sample(self.memory, k=self.batch_size)
# #         states = torch.from_numpy(
# #             np.vstack([e.state for e in exps]).astype(np.float32)
# #         ).to(DEVICE)
# #         actions = torch.from_numpy(
# #             np.vstack([e.action for e in exps]).astype(np.float32)
# #         ).to(DEVICE)
# #         rewards = torch.from_numpy(
# #             np.vstack([e.reward for e in exps]).astype(np.float32)
# #         ).to(DEVICE)
# #         next_states = torch.from_numpy(
# #             np.vstack([e.next_state for e in exps]).astype(np.float32)
# #         ).to(DEVICE)
# #         dones = torch.from_numpy(
# #             np.vstack([e.done for e in exps]).astype(np.uint8)
# #         ).float().to(DEVICE)
# #         return (states, actions, rewards, next_states, dones)

# #     def __len__(self):
# #         return len(self.memory)


# # # ---------------------------------- Ornstein–Uhlenbeck Noise ------------------------------
# # class OUNoise:
# #     def __init__(self, size, seed=0, mu=0.0, theta=0.15, sigma=0.2):
# #         self.mu = mu * np.ones(size, dtype=np.float32)
# #         self.theta = theta
# #         self.sigma = sigma
# #         self.state = self.mu.copy()
# #         random.seed(seed)
# #         np.random.seed(seed)

# #     def reset(self):
# #         self.state = self.mu.copy()

# #     def sample(self):
# #         dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(len(self.state))
# #         self.state += dx
# #         return self.state


# # # ------------------------------------------- Agent ----------------------------------------
# # class Agent:
# #     """
# #     DDPG Agent for Udacity Reacher (single or 20-agent training loop).
# #     """

# #     def __init__(
# #         self,
# #         state_size,
# #         action_size,
# #         random_seed=2,
# #         # Tuned/stable defaults
# #         actor_lr=1e-4,          # ↑ a touch vs 5e-5 for better signal
# #         critic_lr=1e-3,
# #         weight_decay=1e-2,
# #         gamma=0.99,
# #         tau=1e-3,
# #         batch_size=256,
# #         buffer_size=int(1e6),
# #         start_learn_after=5000, # train_20.py passes 1000 for multi-agent
# #         updates_per_step=1,
# #         noise_sigma=0.15,
# #     ):
# #         self.state_size = state_size
# #         self.action_size = action_size
# #         self.gamma = gamma
# #         self.tau = tau
# #         self.batch_size = batch_size
# #         self.updates_per_step = updates_per_step
# #         self.start_learn_after = start_learn_after

# #         torch.manual_seed(random_seed)
# #         np.random.seed(random_seed)
# #         random.seed(random_seed)

# #         # Running state normalization
# #         self.state_norm = RunningNorm(state_size)

# #         # Actor
# #         self.actor_local = Actor(state_size, action_size, seed=random_seed).to(DEVICE)
# #         self.actor_target = Actor(state_size, action_size, seed=random_seed).to(DEVICE)
# #         self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=actor_lr)

# #         # Critic
# #         self.critic_local = Critic(state_size, action_size, seed=random_seed).to(DEVICE)
# #         self.critic_target = Critic(state_size, action_size, seed=random_seed).to(DEVICE)
# #         self.critic_optimizer = optim.Adam(
# #             self.critic_local.parameters(), lr=critic_lr, weight_decay=weight_decay
# #         )

# #         # Initialize target = local
# #         self.hard_update(self.actor_target, self.actor_local)
# #         self.hard_update(self.critic_target, self.critic_local)

# #         # Replay buffer & noise
# #         self.memory = ReplayBuffer(
# #             action_size, buffer_size=buffer_size, batch_size=batch_size, seed=random_seed
# #         )
# #         self.noise = OUNoise(action_size, seed=random_seed, sigma=noise_sigma)
# #         self.noise_scale = 1.0  # gentle decay over episodes

# #     # ------------------------------- Utils -------------------------------
# #     def hard_update(self, target, source):
# #         for t, s in zip(target.parameters(), source.parameters()):
# #             t.data.copy_(s.data)

# #     @staticmethod
# #     def soft_update(local_model, target_model, tau):
# #         for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
# #             target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

# #     # ------------------------------- Agent API ---------------------------
# #     def step(self, state, action, reward, next_state, done, t):
# #         """
# #         Single-agent training path: add one transition and maybe learn.
# #         (In 20-agent training we add transitions externally and call learn once per env step.)
# #         """
# #         self.memory.add(state, action, reward, next_state, done)
# #         if len(self.memory) > self.start_learn_after:
# #             for _ in range(self.updates_per_step):
# #                 experiences = self.memory.sample()
# #                 self.learn(experiences)

# #     def act(self, state, add_noise=True):
# #         """
# #         Returns actions for given state as per current policy.
# #         Applies running normalization to the state.
# #         """
# #         # Update running stats with streaming state & normalize for inference
# #         self.state_norm.update(state)
# #         nstate = self.state_norm.normalize(state)

# #         state_t = torch.from_numpy(nstate).float().unsqueeze(0).to(DEVICE)
# #         self.actor_local.eval()
# #         with torch.no_grad():
# #             action = self.actor_local(state_t).cpu().numpy().squeeze(0)
# #         self.actor_local.train()

# #         if add_noise:
# #             action = action + self.noise_scale * self.noise.sample()
# #         return np.clip(action, -1.0, 1.0)

# #     def reset(self):
# #         self.noise.reset()
# #         # gentle noise decay
# #         self.noise_scale = max(0.05, self.noise_scale * 0.999)

# #     def learn(self, experiences):
# #         """
# #         Update actor and critic using sampled batch of experiences.
# #         Applies normalization to states (but does NOT update running stats here).
# #         """
# #         states, actions, rewards, next_states, dones = experiences

# #         # Normalize states via running stats (no stat updates here)
# #         with torch.no_grad():
# #             s_cpu = states.detach().cpu().numpy()
# #             ns_cpu = next_states.detach().cpu().numpy()
# #             s_n = torch.from_numpy(self.state_norm.normalize(s_cpu)).float().to(DEVICE)
# #             ns_n = torch.from_numpy(self.state_norm.normalize(ns_cpu)).float().to(DEVICE)

# #         # ----- Critic update -----
# #         with torch.no_grad():
# #             next_actions = self.actor_target(ns_n)
# #             Q_targets_next = self.critic_target(ns_n, next_actions)
# #             Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

# #         Q_expected = self.critic_local(s_n, actions)
# #         critic_loss = nn.MSELoss()(Q_expected, Q_targets)
# #         self.critic_optimizer.zero_grad()
# #         critic_loss.backward()
# #         nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1.0)
# #         self.critic_optimizer.step()

# #         # ----- Actor update -----
# #         actions_pred = self.actor_local(s_n)
# #         actor_loss = -self.critic_local(s_n, actions_pred).mean()
# #         self.actor_optimizer.zero_grad()
# #         actor_loss.backward()
# #         nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1.0)  # clip actor too
# #         self.actor_optimizer.step()

# #         # ----- Soft update -----
# #         self.soft_update(self.critic_local, self.critic_target, self.tau)
# #         self.soft_update(self.actor_local, self.actor_target, self.tau)



# # # # coursework_2/ddpg_agent.py
# # # import random
# # # import numpy as np
# # # from collections import deque, namedtuple

# # # import torch
# # # import torch.nn as nn
# # # import torch.optim as optim

# # # from model import Actor, Critic

# # # DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # # class ReplayBuffer:
# # #     def __init__(self, action_size, buffer_size=int(1e6), batch_size=128, seed=0):
# # #         self.action_size = action_size
# # #         self.memory = deque(maxlen=buffer_size)
# # #         self.batch_size = batch_size
# # #         self.experience = namedtuple("Experience",
# # #                                      field_names=["state","action","reward","next_state","done"])
# # #         random.seed(seed)
# # #         np.random.seed(seed)

# # #     def add(self, state, action, reward, next_state, done):
# # #         e = self.experience(state, action, reward, next_state, done)
# # #         self.memory.append(e)

# # #     def sample(self):
# # #         exps = random.sample(self.memory, k=self.batch_size)
# # #         states      = torch.from_numpy(np.vstack([e.state for e in exps])).float().to(DEVICE)
# # #         actions     = torch.from_numpy(np.vstack([e.action for e in exps])).float().to(DEVICE)
# # #         rewards     = torch.from_numpy(np.vstack([e.reward for e in exps])).float().to(DEVICE)
# # #         next_states = torch.from_numpy(np.vstack([e.next_state for e in exps])).float().to(DEVICE)
# # #         dones       = torch.from_numpy(np.vstack([e.done for e in exps]).astype(np.uint8)).float().to(DEVICE)
# # #         return (states, actions, rewards, next_states, dones)

# # #     def __len__(self):
# # #         return len(self.memory)

# # # class OUNoise:
# # #     """Ornstein–Uhlenbeck noise for exploration."""
# # #     def __init__(self, size, seed=0, mu=0.0, theta=0.15, sigma=0.2):
# # #         self.mu = mu * np.ones(size)
# # #         self.theta = theta
# # #         self.sigma = sigma
# # #         self.state = self.mu.copy()
# # #         random.seed(seed)
# # #         np.random.seed(seed)

# # #     def reset(self):
# # #         self.state = self.mu.copy()

# # #     def sample(self):
# # #         dx = self.theta*(self.mu - self.state) + self.sigma*np.random.randn(len(self.state))
# # #         self.state += dx
# # #         return self.state

# # # class Agent:
# # #     """
# # #     DDPG Agent for single-agent Reacher.
# # #     """

# # #     #first attempt -- really really bad
# # #     # def __init__(self, state_size, action_size, random_seed=2,
# # #     #              actor_lr=1e-4, critic_lr=1e-3, weight_decay=1e-2,
# # #     #              gamma=0.99, tau=1e-3, batch_size=128, buffer_size=int(1e6),
# # #     #              start_learn_after=5000, updates_per_step=1, noise_sigma=0.2):
# # #     #     self.state_size  = state_size
# # #     #     self.action_size = action_size
# # #     #     self.gamma = gamma
# # #     #     self.tau = tau
# # #     #     self.batch_size = batch_size
# # #     #     self.updates_per_step = updates_per_step
# # #     #     self.start_learn_after = start_learn_after

# # #     #     torch.manual_seed(random_seed)
# # #     #     np.random.seed(random_seed)
# # #     #     random.seed(random_seed)

# # #     #     # Actor
# # #     #     self.actor_local  = Actor(state_size, action_size, seed=random_seed).to(DEVICE)
# # #     #     self.actor_target = Actor(state_size, action_size, seed=random_seed).to(DEVICE)
# # #     #     self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=actor_lr)

# # #     #     # Critic
# # #     #     self.critic_local  = Critic(state_size, action_size, seed=random_seed).to(DEVICE)
# # #     #     self.critic_target = Critic(state_size, action_size, seed=random_seed).to(DEVICE)
# # #     #     self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
# # #     #                                        lr=critic_lr, weight_decay=weight_decay)

# # #     #     # Initialize target = local
# # #     #     self.hard_update(self.actor_target, self.actor_local)
# # #     #     self.hard_update(self.critic_target, self.critic_local)

# # #     #     # Replay buffer & noise
# # #     #     self.memory = ReplayBuffer(action_size, buffer_size=buffer_size, batch_size=batch_size, seed=random_seed)
# # #     #     self.noise = OUNoise(action_size, seed=random_seed, sigma=noise_sigma)

# # #     # #still bad but less bad than before
# # #     # def __init__(self, state_size, action_size, random_seed=2,
# # #     #              # === Tuned defaults ===
# # #     #              actor_lr=5e-5,             # ↓ was 1e-4
# # #     #              critic_lr=1e-3,
# # #     #              weight_decay=1e-2,
# # #     #              gamma=0.99,
# # #     #              tau=1e-3,
# # #     #              batch_size=256,            # ↑ was 128
# # #     #              buffer_size=int(1e6),
# # #     #              start_learn_after=10000,   # ↑ gives more diverse buffer before learning
# # #     #              updates_per_step=1,
# # #     #              noise_sigma=0.15):         # ↓ was 0.2
# # #     #     self.state_size  = state_size
# # #     #     self.action_size = action_size
# # #     #     self.gamma = gamma
# # #     #     self.tau = tau
# # #     #     self.batch_size = batch_size
# # #     #     self.updates_per_step = updates_per_step
# # #     #     self.start_learn_after = start_learn_after

# # #     #     torch.manual_seed(random_seed)
# # #     #     np.random.seed(random_seed)
# # #     #     random.seed(random_seed)

# # #     #     # Actor
# # #     #     self.actor_local  = Actor(state_size, action_size, seed=random_seed).to(DEVICE)
# # #     #     self.actor_target = Actor(state_size, action_size, seed=random_seed).to(DEVICE)
# # #     #     self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=actor_lr)

# # #     #     # Critic
# # #     #     self.critic_local  = Critic(state_size, action_size, seed=random_seed).to(DEVICE)
# # #     #     self.critic_target = Critic(state_size, action_size, seed=random_seed).to(DEVICE)
# # #     #     self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
# # #     #                                        lr=critic_lr, weight_decay=weight_decay)

# # #     #     # Initialize target = local
# # #     #     self.hard_update(self.actor_target, self.actor_local)
# # #     #     self.hard_update(self.critic_target, self.critic_local)

# # #     #     # Replay buffer & noise
# # #     #     self.memory = ReplayBuffer(action_size, buffer_size=buffer_size, batch_size=batch_size, seed=random_seed)
# # #     #     self.noise = OUNoise(action_size, seed=random_seed, sigma=noise_sigma)

# # #     def __init__(self, state_size, action_size, random_seed=2,
# # #                  # Tuned defaults + stronger updates
# # #                  actor_lr=5e-5,
# # #                  critic_lr=1e-3,
# # #                  weight_decay=1e-2,
# # #                  gamma=0.99,
# # #                  tau=1e-3,
# # #                  batch_size=256,
# # #                  buffer_size=int(1e6),
# # #                  start_learn_after=5000,    # ↓ was 10000
# # #                  updates_per_step=2,        # ↑ was 1
# # #                  noise_sigma=0.15):
# # #         self.state_size  = state_size
# # #         self.action_size = action_size
# # #         self.gamma = gamma
# # #         self.tau = tau
# # #         self.batch_size = batch_size
# # #         self.updates_per_step = updates_per_step
# # #         self.start_learn_after = start_learn_after

# # #         torch.manual_seed(random_seed)
# # #         np.random.seed(random_seed)
# # #         random.seed(random_seed)

# # #         # Actor
# # #         self.actor_local  = Actor(state_size, action_size, seed=random_seed).to(DEVICE)
# # #         self.actor_target = Actor(state_size, action_size, seed=random_seed).to(DEVICE)
# # #         self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=actor_lr)

# # #         # Critic
# # #         self.critic_local  = Critic(state_size, action_size, seed=random_seed).to(DEVICE)
# # #         self.critic_target = Critic(state_size, action_size, seed=random_seed).to(DEVICE)
# # #         self.critic_optimizer = optim.Adam(
# # #             self.critic_local.parameters(),
# # #             lr=critic_lr, weight_decay=weight_decay
# # #         )

# # #         # Initialize target = local
# # #         self.hard_update(self.actor_target, self.actor_local)
# # #         self.hard_update(self.critic_target, self.critic_local)

# # #         # Replay buffer & noise
# # #         self.memory = ReplayBuffer(
# # #             action_size, buffer_size=buffer_size, batch_size=batch_size, seed=random_seed
# # #         )
# # #         self.noise = OUNoise(action_size, seed=random_seed, sigma=noise_sigma)

# # #         # Optional: gentle exploration decay over time
# # #         self.noise_scale = 1.0  # will decay a bit each episode


# # #     def hard_update(self, target, source):
# # #         for t, s in zip(target.parameters(), source.parameters()):
# # #             t.data.copy_(s.data)

# # #     def step(self, state, action, reward, next_state, done, t):
# # #         self.memory.add(state, action, reward, next_state, done)
# # #         # Learn
# # #         if len(self.memory) > self.start_learn_after:
# # #             for _ in range(self.updates_per_step):
# # #                 experiences = self.memory.sample()
# # #                 self.learn(experiences)

# # #     def act(self, state, add_noise=True):
# # #         state_t = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
# # #         self.actor_local.eval()
# # #         with torch.no_grad():
# # #             action = self.actor_local(state_t).cpu().data.numpy().squeeze(0)
# # #         self.actor_local.train()
# # #         # if add_noise:
# # #         #     action += self.noise.sample()
# # #         # Hopefully this will improve
# # #         if add_noise:
# # #             action += self.noise_scale * self.noise.sample()            
# # #         return np.clip(action, -1, 1)

# # #     # def reset(self):
# # #     #     self.noise.reset()

# # #     # this is better
# # #     def reset(self):
# # #         self.noise.reset()
# # #         self.noise_scale = max(0.05, self.noise_scale * 0.999)  # gentle decay


# # #     def learn(self, experiences):
# # #         states, actions, rewards, next_states, dones = experiences

# # #         # ----- Critic update -----
# # #         with torch.no_grad():
# # #             next_actions = self.actor_target(next_states)
# # #             Q_targets_next = self.critic_target(next_states, next_actions)
# # #             Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
# # #         Q_expected = self.critic_local(states, actions)
# # #         critic_loss = nn.MSELoss()(Q_expected, Q_targets)
# # #         self.critic_optimizer.zero_grad()
# # #         critic_loss.backward()
# # #         nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1.0)
# # #         self.critic_optimizer.step()

# # #         # ----- Actor update -----
# # #         actions_pred = self.actor_local(states)
# # #         actor_loss = -self.critic_local(states, actions_pred).mean()
# # #         self.actor_optimizer.zero_grad()
# # #         actor_loss.backward()
# # #         self.actor_optimizer.step()

# # #         # ----- Soft update -----
# # #         self.soft_update(self.critic_local, self.critic_target, self.tau)
# # #         self.soft_update(self.actor_local,  self.actor_target,  self.tau)

# # #     @staticmethod
# # #     def soft_update(local_model, target_model, tau):
# # #         for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
# # #             target_param.data.copy_(tau*local_param.data + (1.0 - tau)*target_param.data)


