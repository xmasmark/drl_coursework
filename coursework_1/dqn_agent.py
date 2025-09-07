import random, collections, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# # ----- Networks -----
# class QNetworkMLP(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden=(128,128)):
#         super().__init__()
#         layers = []
#         dims = (state_dim,)+hidden+(action_dim,)
#         for i in range(len(dims)-2):
#             layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU()]
#         layers += [nn.Linear(dims[-2], dims[-1])]
#         self.net = nn.Sequential(*layers)
#     def forward(self, x): return self.net(x.float())

# class QNetworkCNN(nn.Module):
#     # Use only if your state is 84x84x4 like Atari; otherwise stick to MLP.
#     def __init__(self, action_dim):
#         super().__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
#             nn.Conv2d(32,64,4, stride=2), nn.ReLU(),
#             nn.Conv2d(64,64,3, stride=1), nn.ReLU(),
#         )
#         self.head = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(64*7*7, 512), nn.ReLU(),
#             nn.Linear(512, action_dim),
#         )
#     def forward(self, x):
#         # x: (B, 4, 84, 84)
#         return self.head(self.features(x.float()/255.0))

# # ----- Replay Buffer -----
# class ReplayBuffer:
#     def __init__(self, capacity=100_000):
#         self.buf = collections.deque(maxlen=capacity)
#     def push(self, s,a,r,s2,d):
#         self.buf.append((s,a,r,s2,d))
#     def sample(self, batch_size):
#         batch = random.sample(self.buf, batch_size)
#         s,a,r,s2,d = map(np.array, zip(*batch))
#         return s,a,r,s2,d
#     def __len__(self): return len(self.buf)



# class DQNAgent:
#     def __init__(self, state_dim, action_dim, device=None,
#                  model_type="mlp", hidden=(128,128),
#                  gamma=0.99, lr=6.25e-4,
#                  eps_start=1.0, eps_end=0.05, eps_decay_steps=50_000,
#                  target_update_every=1_000, batch_size=64,
#                  replay_capacity=100_000, clip_reward=False, clip_td_error=True):
#         self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
#         # ... rest of your __init__ exactly as you have it ...

#     def epsilon(self):
#         frac = min(1.0, self.total_steps / self.eps_decay_steps)
#         return self.eps_start + frac * (self.eps_end - self.eps_start)

#     @torch.no_grad()
#     def act(self, state, eps=None):
#         """Udacity-compatible: if eps is provided, use it; else use internal schedule."""
#         self.total_steps += 1
#         use_eps = self.epsilon() if eps is None else eps
#         if random.random() < use_eps:
#             return random.randrange(self.action_dim)
#         state_t = torch.as_tensor(state, device=self.device).unsqueeze(0)
#         qvals = self.q(state_t)
#         return int(torch.argmax(qvals, dim=1).item())

#     # --- Udacity-compatible API ---
#     def step(self, state, action, reward, next_state, done):
#         """Mirror Udacity's Agent.step: store transition and (maybe) learn."""
#         self.push(state, action, reward, next_state, done)
#         return self.train_step()

#     # keep your push(...) and train_step(...) as-is

# # Provide a drop-in alias so `from dqn_agent import Agent` works
# Agent = DQNAgent

# dqn_agent.py
import random, collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ===== Networks =====
class QNetworkMLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=(128, 128)):
        super().__init__()
        layers = []
        dims = (state_dim,) + hidden + (action_dim,)
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.float())


class QNetworkCNN(nn.Module):
    """Use only for image input like (4,84,84). Not needed for Banana."""
    def __init__(self, action_dim):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
            nn.Linear(512, action_dim),
        )

    def forward(self, x):
        x = x.float() / 255.0
        return self.head(self.features(x))


# ===== Replay Buffer =====
class ReplayBuffer:
    def __init__(self, capacity=100_000, seed=None):
        self.buf = collections.deque(maxlen=capacity)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def push(self, s, a, r, s2, d):
        self.buf.append((s, a, r, s2, d))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = map(np.array, zip(*batch))
        # ensure correct shapes / dtypes
        a = a.astype(np.int64)
        d = d.astype(np.float32)
        return s, a, r, s2, d

    def __len__(self):
        return len(self.buf)


# ===== Agent =====
class DQNAgent:
    def __init__(
        self,
        # accept both Udacity-style and our style
        state_size=None, action_size=None, seed=None,
        state_dim=None, action_dim=None,
        device=None,
        model_type="mlp", hidden=(128, 128),
        gamma=0.99, lr=6.25e-4,
        eps_start=1.0, eps_end=0.05, eps_decay_steps=50_000,
        target_update_every=1_000, batch_size=64,
        replay_capacity=100_000, clip_reward=False, clip_td_error=True,
        update_every=1  # learn every N env steps (1 ~ Udacity UPDATE_EVERY=4 if you prefer, set 4)
    ):
        # dims (support both styles)
        state_dim = state_dim or state_size
        action_dim = action_dim or action_size
        assert state_dim is not None and action_dim is not None, "Provide state_dim/state_size and action_dim/action_size"

        # device & seeds
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        if seed is not None:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

        # config
        self.action_dim = int(action_dim)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.target_update_every = int(target_update_every)
        self.clip_reward = bool(clip_reward)
        self.clip_td_error = bool(clip_td_error)
        self.update_every = int(update_every)

        # epsilon schedule
        self.eps_start = float(eps_start)
        self.eps_end = float(eps_end)
        self.eps_decay_steps = int(eps_decay_steps)

        # counters
        self.total_steps = 0          # env steps (used for epsilon + target sync cadence)
        self.learn_steps = 0          # gradient steps (for target sync if you prefer)
        self._since_update = 0

        # networks
        if model_type == "mlp":
            self.q = QNetworkMLP(state_dim, action_dim, hidden).to(self.device)
            self.q_tgt = QNetworkMLP(state_dim, action_dim, hidden).to(self.device)
        else:
            self.q = QNetworkCNN(action_dim).to(self.device)
            self.q_tgt = QNetworkCNN(action_dim).to(self.device)
        self.q_tgt.load_state_dict(self.q.state_dict())
        self.q_tgt.eval()

        # optimizer (Nature DQN used RMSProp)
        self.opt = optim.RMSprop(self.q.parameters(), lr=lr, eps=1e-5, alpha=0.95)

        # replay
        self.replay = ReplayBuffer(replay_capacity, seed=seed)

        # loss (Huber-like via manual clip, or SmoothL1Loss)
        self._mse = nn.MSELoss()
        self._huber = nn.SmoothL1Loss()

    # ----- Epsilon schedule -----
    def epsilon(self):
        frac = min(1.0, self.total_steps / max(1, self.eps_decay_steps))
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    @torch.no_grad()
    def act(self, state, eps=None):
        """Udacity-compatible: if eps is provided, use it; else use internal schedule."""
        self.total_steps += 1
        use_eps = self.epsilon() if eps is None else float(eps)
        if random.random() < use_eps:
            return random.randrange(self.action_dim)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        qvals = self.q(state_t)
        return int(torch.argmax(qvals, dim=1).item())

    # ----- Udacity-compatible API -----
    def step(self, state, action, reward, next_state, done):
        """Store transition and (optionally) learn."""
        if self.clip_reward:
            reward = max(-1.0, min(1.0, float(reward)))
        self.replay.push(state, action, reward, next_state, done)

        self._since_update += 1
        if self._since_update >= self.update_every:
            self._since_update = 0
            return self.train_step()
        return None

    # ----- One optimization step -----
    def train_step(self):
        if len(self.replay) < self.batch_size:
            return None

        s, a, r, s2, d = self.replay.sample(self.batch_size)
        s_t  = torch.as_tensor(s,  dtype=torch.float32, device=self.device)
        a_t  = torch.as_tensor(a,  dtype=torch.long,   device=self.device).unsqueeze(1)
        r_t  = torch.as_tensor(r,  dtype=torch.float32, device=self.device).unsqueeze(1)
        s2_t = torch.as_tensor(s2, dtype=torch.float32, device=self.device)
        d_t  = torch.as_tensor(d,  dtype=torch.float32, device=self.device).unsqueeze(1)

        # Q(s,a)
        q_sa = self.q(s_t).gather(1, a_t)

        with torch.no_grad():
            q_s2_max = self.q_tgt(s2_t).max(1, keepdim=True)[0]
            y = r_t + self.gamma * (1.0 - d_t) * q_s2_max

        td = y - q_sa
        if self.clip_td_error:
            # manual clip like DQN’s stability trick
            td_clipped = td.clamp(-1.0, 1.0)
            loss = (td_clipped ** 2).mean()
        else:
            loss = self._huber(q_sa, y)  # or self._mse

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()

        self.learn_steps += 1
        # sync target every N gradient steps (or you can use total env steps)
        if self.learn_steps % self.target_update_every == 0:
            self.q_tgt.load_state_dict(self.q.state_dict())

        return float(loss.item())


# Alias for Udacity import style
Agent = DQNAgent
