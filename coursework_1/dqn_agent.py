import random, collections, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ----- Networks -----
class QNetworkMLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=(128,128)):
        super().__init__()
        layers = []
        dims = (state_dim,)+hidden+(action_dim,)
        for i in range(len(dims)-2):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU()]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x.float())

class QNetworkCNN(nn.Module):
    # Use only if your state is 84x84x4 like Atari; otherwise stick to MLP.
    def __init__(self, action_dim):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32,64,4, stride=2), nn.ReLU(),
            nn.Conv2d(64,64,3, stride=1), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7, 512), nn.ReLU(),
            nn.Linear(512, action_dim),
        )
    def forward(self, x):
        # x: (B, 4, 84, 84)
        return self.head(self.features(x.float()/255.0))

# ----- Replay Buffer -----
class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buf = collections.deque(maxlen=capacity)
    def push(self, s,a,r,s2,d):
        self.buf.append((s,a,r,s2,d))
    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s,a,r,s2,d = map(np.array, zip(*batch))
        return s,a,r,s2,d
    def __len__(self): return len(self.buf)

# ----- Agent -----
class DQNAgent:
    def __init__(
        self, state_dim, action_dim, device="cpu",
        model_type="mlp", hidden=(128,128),
        gamma=0.99, lr=6.25e-4,  # RMSProp-friendly LR
        eps_start=1.0, eps_end=0.05, eps_decay_steps=50_000,
        target_update_every=1_000, batch_size=64,
        replay_capacity=100_000, clip_reward=False, clip_td_error=True
    ):
        self.device = torch.device(device)
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_every = target_update_every
        self.clip_reward = clip_reward
        self.clip_td_error = clip_td_error

        if model_type == "mlp":
            self.q = QNetworkMLP(state_dim, action_dim, hidden).to(self.device)
            self.q_tgt = QNetworkMLP(state_dim, action_dim, hidden).to(self.device)
        else:
            self.q = QNetworkCNN(action_dim).to(self.device)
            self.q_tgt = QNetworkCNN(action_dim).to(self.device)

        self.q_tgt.load_state_dict(self.q.state_dict())
        # Paper used RMSProp; keep it for fidelity
        self.opt = optim.RMSprop(self.q.parameters(), lr=lr, eps=1e-5, alpha=0.95)
        self.replay = ReplayBuffer(replay_capacity)

        # ε schedule
        self.eps_start, self.eps_end = eps_start, eps_end
        self.eps_decay_steps = eps_decay_steps
        self.total_steps = 0
        self.loss_fn = nn.SmoothL1Loss() if clip_td_error else nn.MSELoss()

    def epsilon(self):
        # linear decay
        frac = min(1.0, self.total_steps / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    @torch.no_grad()
    def act(self, state):
        self.total_steps += 1
        if random.random() < self.epsilon():
            return random.randrange(self.action_dim)
        state_t = torch.as_tensor(state, device=self.device).unsqueeze(0)
        qvals = self.q(state_t)
        return int(torch.argmax(qvals, dim=1).item())

    def push(self, *transition):  # s,a,r,s2,d
        s,a,r,s2,d = transition
        if self.clip_reward:
            r = max(-1.0, min(1.0, r))
        self.replay.push(s,a,r,s2,d)

    def train_step(self):
        if len(self.replay) < self.batch_size: return None
        s,a,r,s2,d = self.replay.sample(self.batch_size)

        s_t  = torch.as_tensor(s,  device=self.device)
        a_t  = torch.as_tensor(a,  device=self.device).long().unsqueeze(1)
        r_t  = torch.as_tensor(r,  device=self.device).float().unsqueeze(1)
        s2_t = torch.as_tensor(s2, device=self.device)
        d_t  = torch.as_tensor(d,  device=self.device).float().unsqueeze(1)

        # Q(s,a)
        q_sa = self.q(s_t).gather(1, a_t)

        with torch.no_grad():
            # max_a' Q_tgt(s', a')
            q_s2_max = self.q_tgt(s2_t).max(1, keepdim=True)[0]
            y = r_t + self.gamma * (1.0 - d_t) * q_s2_max

        td = y - q_sa
        if self.clip_td_error:
            td = td.clamp(-1, 1)  # Huber-like clipping as per paper’s stability trick
            loss = (td**2).mean()
        else:
            loss = self.loss_fn(q_sa, y)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()

        if self.total_steps % self.target_update_every == 0:
            self.q_tgt.load_state_dict(self.q.state_dict())
        return float(loss.item())
