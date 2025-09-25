# maddpg_agent.py
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic
from replay_buffer import ReplayBuffer
from noise import OUNoise

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    """Single agent with decentralized actor and centralized critic."""

    def __init__(self, idx, state_size, action_size, n_agents, seed=0,
                 lr_actor=1e-4, lr_critic=1e-3, weight_decay=1e-5):
        self.idx = idx
        self.n_agents = n_agents
        self.state_size = state_size
        self.action_size = action_size

        # Actor (local + target)
        self.actor_local = Actor(state_size, action_size, seed).to(DEVICE)
        self.actor_target = Actor(state_size, action_size, seed + 1).to(DEVICE)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)

        # Centralized critic sees all states and all actions
        full_state_size = n_agents * state_size
        full_action_size = n_agents * action_size

        self.critic_local = Critic(full_state_size, full_action_size, seed).to(DEVICE)
        self.critic_target = Critic(full_state_size, full_action_size, seed + 1).to(DEVICE)
        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(), lr=lr_critic, weight_decay=weight_decay
        )

        # Init target = local
        self.hard_update(self.actor_target, self.actor_local)
        self.hard_update(self.critic_target, self.critic_local)

        # OU exploration noise (per-agent)
        self.noise = OUNoise((action_size,), seed=seed)

    def act(self, state, add_noise=True):
        """Select action for a single agent."""
        state = torch.from_numpy(state).float().to(DEVICE).unsqueeze(0)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy().squeeze(0)
        self.actor_local.train()
        if add_noise:
            action = np.clip(action + self.noise.sample(), -1, 1)
        return action

    def reset(self):
        self.noise.reset()

    @staticmethod
    def soft_update(local_model, target_model, tau):
        for tp, lp in zip(target_model.parameters(), local_model.parameters()):
            tp.data.copy_(tau * lp.data + (1.0 - tau) * tp.data)

    @staticmethod
    def hard_update(target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(s.data)


class MADDPG:
    """Multi-agent DDPG with centralized critics and shared replay."""

    def __init__(self, state_size, action_size, n_agents=2, seed=0,
                 gamma=0.99, tau=1e-3, lr_actor=1e-4, lr_critic=1e-3,
                 weight_decay=1e-5, buffer_size=int(1e6), batch_size=256,
                 updates_per_step=4):
        self.n_agents = n_agents
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.updates_per_step = updates_per_step

        # Build agents
        self.agents = [
            Agent(i, state_size, action_size, n_agents, seed + i,
                  lr_actor, lr_critic, weight_decay)
            for i in range(n_agents)
        ]

        # Shared replay
        self.memory = ReplayBuffer(buffer_size, batch_size, seed, DEVICE)

    def act(self, states, add_noise=True):
        """States: array/list of shape (n_agents, state_size)."""
        return np.vstack([ag.act(states[i], add_noise) for i, ag in enumerate(self.agents)])

    def step(self, states, actions, rewards, next_states, dones):
        """
        Store full joint transition; rewards/dones are vectors (per-agent).
        - states, next_states: (n_agents, state_size)
        - actions: (n_agents, action_size)
        - rewards, dones: (n_agents,)
        """
        full_s = np.concatenate(states)       # (S_all,)
        full_ns = np.concatenate(next_states) # (S_all,)
        full_a = np.concatenate(actions)      # (A_all,)

        # ✅ Store per-agent reward & done vectors (no averaging)
        self.memory.add(
            full_s,
            full_a,
            np.array(rewards, dtype=np.float32),
            full_ns,
            np.array(dones, dtype=np.uint8),
        )

        # Learn
        if len(self.memory) > self.batch_size:
            for _ in range(self.updates_per_step):
                self.learn()

    def learn(self):
        """Sample a batch and update critics then actors."""
        states, actions, rewards, next_states, dones = self.memory.sample()
        # Chunk into per-agent views
        ss = torch.chunk(states, self.n_agents, dim=1)       # list of (B, state_size)
        ns = torch.chunk(next_states, self.n_agents, dim=1)  # list of (B, state_size)

        # Next joint actions from target policies
        next_actions_list = [ag.actor_target(ns[i]) for i, ag in enumerate(self.agents)]
        next_actions = torch.cat(next_actions_list, dim=1)    # (B, A_all)

        # ----- Update each critic (agent-specific reward/done) -----
        for i, ag in enumerate(self.agents):
            r_i = rewards[:, i].unsqueeze(1)  # (B,1)
            d_i = dones[:, i].unsqueeze(1)    # (B,1)

            q_targets_next = ag.critic_target(next_states, next_actions)
            q_targets = r_i + (self.gamma * q_targets_next * (1 - d_i))

            q_expected = ag.critic_local(states, actions)

            critic_loss = F.mse_loss(q_expected, q_targets.detach())
            ag.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(ag.critic_local.parameters(), 1.0)
            ag.critic_optimizer.step()

        # ----- Update each actor -----
        for i, ag in enumerate(self.agents):
            # Build joint actions where agent i uses its current actor (others detached)
            actions_pred_list = []
            for j, other in enumerate(self.agents):
                a_j = other.actor_local(ss[j])
                actions_pred_list.append(a_j if j == i else a_j.detach())
            actions_pred = torch.cat(actions_pred_list, dim=1)
            actor_loss = -ag.critic_local(states, actions_pred).mean()
            ag.actor_optimizer.zero_grad()
            actor_loss.backward()
            ag.actor_optimizer.step()

        # Soft-update all targets
        for ag in self.agents:
            Agent.soft_update(ag.actor_local, ag.actor_target, self.tau)
            Agent.soft_update(ag.critic_local, ag.critic_target, self.tau)

    def reset(self):
        for ag in self.agents:
            ag.reset()

    # Optional helper if you want to decay exploration noise from train.py
    def set_noise_sigma(self, sigma: float):
        for ag in self.agents:
            ag.noise.sigma = float(sigma)



# import copy
# import numpy as np
# import torch
# import torch.nn.functional as F
# import torch.optim as optim
# from model import Actor, Critic
# from replay_buffer import ReplayBuffer
# from noise import OUNoise

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# class Agent:
#     def __init__(self, idx, state_size, action_size, n_agents, seed=0,
#                  lr_actor=1e-3, lr_critic=1e-3, weight_decay=0.0):
#         self.idx = idx
#         self.n_agents = n_agents
#         self.state_size = state_size
#         self.action_size = action_size

#         self.actor_local = Actor(state_size, action_size, seed).to(DEVICE)
#         self.actor_target = Actor(state_size, action_size, seed+1).to(DEVICE)
#         self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)

#         full_state_size = n_agents * state_size
#         full_action_size = n_agents * action_size

#         self.critic_local = Critic(full_state_size, full_action_size, seed).to(DEVICE)
#         self.critic_target = Critic(full_state_size, full_action_size, seed+1).to(DEVICE)
#         self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=weight_decay)

#         self.hard_update(self.actor_target, self.actor_local)
#         self.hard_update(self.critic_target, self.critic_local)

#         self.noise = OUNoise((action_size,), seed=seed)

#     def act(self, state, add_noise=True):
#         state = torch.from_numpy(state).float().to(DEVICE).unsqueeze(0)
#         self.actor_local.eval()
#         with torch.no_grad():
#             action = self.actor_local(state).cpu().data.numpy().squeeze(0)
#         self.actor_local.train()
#         if add_noise:
#             action += self.noise.sample()
#         return np.clip(action, -1, 1)

#     def reset(self):
#         self.noise.reset()

#     @staticmethod
#     def soft_update(local_model, target_model, tau):
#         for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
#             target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

#     @staticmethod
#     def hard_update(target, source):
#         for t, s in zip(target.parameters(), source.parameters()):
#             t.data.copy_(s.data)

# class MADDPG:
#     # def __init__(self, state_size, action_size, n_agents=2, seed=0,
#     #              gamma=0.99, tau=1e-3, lr_actor=1e-3, lr_critic=1e-3,
#     #              weight_decay=0.0, buffer_size=int(1e6), batch_size=256,
#     #              updates_per_step=2):
#     def __init__(self, state_size, action_size, n_agents=2, seed=0,
#                  gamma=0.99, tau=1e-3, lr_actor=1e-4, lr_critic=1e-3,
#                  weight_decay=1e-5, buffer_size=int(1e6), batch_size=256,
#                  updates_per_step=4):

#         self.n_agents = n_agents
#         self.gamma = gamma
#         self.tau = tau
#         self.batch_size = batch_size
#         self.updates_per_step = updates_per_step

#         self.agents = [Agent(i, state_size, action_size, n_agents, seed+i,
#                              lr_actor, lr_critic, weight_decay) for i in range(n_agents)]

#         self.memory = ReplayBuffer(buffer_size, batch_size, seed, DEVICE)

#     def act(self, states, add_noise=True):
#         return np.vstack([ag.act(states[i], add_noise) for i, ag in enumerate(self.agents)])

#     # def step(self, states, actions, rewards, next_states, dones):
#     #     # Flatten per-agent into full vectors
#     #     full_s = np.concatenate(states)
#     #     full_ns = np.concatenate(next_states)
#     #     full_a = np.concatenate(actions)
#     #     self.memory.add(full_s, full_a, np.mean(rewards), full_ns, np.any(dones).astype(np.float32))

#     #     if len(self.memory) > self.batch_size:
#     #         for _ in range(self.updates_per_step):
#     #             self.learn()


#     def step(self, states, actions, rewards, next_states, dones):
#         full_s  = np.concatenate(states)
#         full_ns = np.concatenate(next_states)
#         full_a  = np.concatenate(actions)

#         # ✅ store per-agent reward & per-agent done (vectors), not mean/any
#         self.memory.add(full_s, full_a, np.array(rewards, dtype=np.float32),
#                         full_ns, np.array(dones, dtype=np.uint8))

#         if len(self.memory) > self.batch_size:
#             for _ in range(self.updates_per_step):
#                 self.learn()


#     # def learn(self):
#     #     states, actions, rewards, next_states, dones = self.memory.sample()
#     #     # Decompose to per-agent pieces
#     #     ss = torch.chunk(states, self.n_agents, dim=1)
#     #     ns = torch.chunk(next_states, self.n_agents, dim=1)
#     #     aa = torch.chunk(actions, self.n_agents, dim=1)

#     #     # Next actions from target actors
#     #     next_actions_list = []
#     #     for i, ag in enumerate(self.agents):
#     #         next_actions_list.append(ag.actor_target(ns[i]))
#     #     next_actions = torch.cat(next_actions_list, dim=1)

#     #     # ---------- Update each critic ----------
#     #     full_q_targets = None  # just to keep code readable
#     #     for i, ag in enumerate(self.agents):
#     #         # Critic target: r + γ * Q'(s', a')
#     #         q_targets_next = ag.critic_target(next_states, next_actions)
#     #         q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))
#     #         q_expected = ag.critic_local(states, actions)
#     #         critic_loss = F.mse_loss(q_expected, q_targets.detach())
#     #         ag.critic_optimizer.zero_grad()
#     #         critic_loss.backward()
#     #         torch.nn.utils.clip_grad_norm_(ag.critic_local.parameters(), 1)
#     #         ag.critic_optimizer.step()

#     #     # ---------- Update each actor ----------
#     #     for i, ag in enumerate(self.agents):
#     #         # build joint actions with agent i’s action from its actor_local, others from detatched current actions
#     #         actions_pred_list = []
#     #         for j, other in enumerate(self.agents):
#     #             if j == i:
#     #                 actions_pred_list.append(other.actor_local(ss[j]))
#     #             else:
#     #                 actions_pred_list.append(other.actor_local(ss[j]).detach())
#     #         actions_pred = torch.cat(actions_pred_list, dim=1)
#     #         actor_loss = -ag.critic_local(states, actions_pred).mean()
#     #         ag.actor_optimizer.zero_grad()
#     #         actor_loss.backward()
#     #         ag.actor_optimizer.step()

#     #     # Soft update targets
#     #     for ag in self.agents:
#     #         Agent.soft_update(ag.actor_local, ag.actor_target, self.tau)
#     #         Agent.soft_update(ag.critic_local, ag.critic_target, self.tau)




#     def reset(self):
#         for ag in self.agents:
#             ag.reset()
