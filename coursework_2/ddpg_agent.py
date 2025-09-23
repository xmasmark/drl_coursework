# coursework_2/ddpg_agent.py
import random
import numpy as np
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.optim as optim

from model import Actor, Critic

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, action_size, buffer_size=int(1e6), batch_size=128, seed=0):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                     field_names=["state","action","reward","next_state","done"])
        random.seed(seed)
        np.random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        exps = random.sample(self.memory, k=self.batch_size)
        states      = torch.from_numpy(np.vstack([e.state for e in exps])).float().to(DEVICE)
        actions     = torch.from_numpy(np.vstack([e.action for e in exps])).float().to(DEVICE)
        rewards     = torch.from_numpy(np.vstack([e.reward for e in exps])).float().to(DEVICE)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in exps])).float().to(DEVICE)
        dones       = torch.from_numpy(np.vstack([e.done for e in exps]).astype(np.uint8)).float().to(DEVICE)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

class OUNoise:
    """Ornstein–Uhlenbeck noise for exploration."""
    def __init__(self, size, seed=0, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = self.mu.copy()
        random.seed(seed)
        np.random.seed(seed)

    def reset(self):
        self.state = self.mu.copy()

    def sample(self):
        dx = self.theta*(self.mu - self.state) + self.sigma*np.random.randn(len(self.state))
        self.state += dx
        return self.state

class Agent:
    """
    DDPG Agent for single-agent Reacher.
    """
    # def __init__(self, state_size, action_size, random_seed=2,
    #              actor_lr=1e-4, critic_lr=1e-3, weight_decay=1e-2,
    #              gamma=0.99, tau=1e-3, batch_size=128, buffer_size=int(1e6),
    #              start_learn_after=5000, updates_per_step=1, noise_sigma=0.2):
    #     self.state_size  = state_size
    #     self.action_size = action_size
    #     self.gamma = gamma
    #     self.tau = tau
    #     self.batch_size = batch_size
    #     self.updates_per_step = updates_per_step
    #     self.start_learn_after = start_learn_after

    #     torch.manual_seed(random_seed)
    #     np.random.seed(random_seed)
    #     random.seed(random_seed)

    #     # Actor
    #     self.actor_local  = Actor(state_size, action_size, seed=random_seed).to(DEVICE)
    #     self.actor_target = Actor(state_size, action_size, seed=random_seed).to(DEVICE)
    #     self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=actor_lr)

    #     # Critic
    #     self.critic_local  = Critic(state_size, action_size, seed=random_seed).to(DEVICE)
    #     self.critic_target = Critic(state_size, action_size, seed=random_seed).to(DEVICE)
    #     self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
    #                                        lr=critic_lr, weight_decay=weight_decay)

    #     # Initialize target = local
    #     self.hard_update(self.actor_target, self.actor_local)
    #     self.hard_update(self.critic_target, self.critic_local)

    #     # Replay buffer & noise
    #     self.memory = ReplayBuffer(action_size, buffer_size=buffer_size, batch_size=batch_size, seed=random_seed)
    #     self.noise = OUNoise(action_size, seed=random_seed, sigma=noise_sigma)


    def __init__(self, state_size, action_size, random_seed=2,
                 # === Tuned defaults ===
                 actor_lr=5e-5,             # ↓ was 1e-4
                 critic_lr=1e-3,
                 weight_decay=1e-2,
                 gamma=0.99,
                 tau=1e-3,
                 batch_size=256,            # ↑ was 128
                 buffer_size=int(1e6),
                 start_learn_after=10000,   # ↑ gives more diverse buffer before learning
                 updates_per_step=1,
                 noise_sigma=0.15):         # ↓ was 0.2
        self.state_size  = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.updates_per_step = updates_per_step
        self.start_learn_after = start_learn_after

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

        # Actor
        self.actor_local  = Actor(state_size, action_size, seed=random_seed).to(DEVICE)
        self.actor_target = Actor(state_size, action_size, seed=random_seed).to(DEVICE)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=actor_lr)

        # Critic
        self.critic_local  = Critic(state_size, action_size, seed=random_seed).to(DEVICE)
        self.critic_target = Critic(state_size, action_size, seed=random_seed).to(DEVICE)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                           lr=critic_lr, weight_decay=weight_decay)

        # Initialize target = local
        self.hard_update(self.actor_target, self.actor_local)
        self.hard_update(self.critic_target, self.critic_local)

        # Replay buffer & noise
        self.memory = ReplayBuffer(action_size, buffer_size=buffer_size, batch_size=batch_size, seed=random_seed)
        self.noise = OUNoise(action_size, seed=random_seed, sigma=noise_sigma)

    def hard_update(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(s.data)

    def step(self, state, action, reward, next_state, done, t):
        self.memory.add(state, action, reward, next_state, done)
        # Learn
        if len(self.memory) > self.start_learn_after:
            for _ in range(self.updates_per_step):
                experiences = self.memory.sample()
                self.learn(experiences)

    def act(self, state, add_noise=True):
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state_t).cpu().data.numpy().squeeze(0)
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # ----- Critic update -----
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            Q_targets_next = self.critic_target(next_states, next_actions)
            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic_local(states, actions)
        critic_loss = nn.MSELoss()(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1.0)
        self.critic_optimizer.step()

        # ----- Actor update -----
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----- Soft update -----
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local,  self.actor_target,  self.tau)

    @staticmethod
    def soft_update(local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0 - tau)*target_param.data)
