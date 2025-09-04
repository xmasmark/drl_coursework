import numpy as np, torch, time
from collections import deque
from dqn_agent import DQNAgent

def train(env, state_dim, action_dim, episodes=1000, max_steps=1000,
          solve_avg_reward=None, device="cpu", **agent_kwargs):
    agent = DQNAgent(state_dim, action_dim, device=device, **agent_kwargs)
    scores, ma = [], deque(maxlen=100)

    for ep in range(1, episodes+1):
        s = env.reset()
        ep_reward, losses = 0.0, []
        for t in range(max_steps):
            a = agent.act(s)
            s2, r, done, _ = env.step(a)
            agent.push(s, a, r, s2, done)
            loss = agent.train_step()
            if loss is not None: losses.append(loss)
            s = s2; ep_reward += r
            if done: break

        scores.append(ep_reward); ma.append(ep_reward)
        print(f"Ep {ep:4d} | R={ep_reward:7.2f} | MA100={np.mean(ma):7.2f} | "
              f"steps={agent.total_steps} | loss={np.mean(losses) if losses else np.nan:.4f}")

        if solve_avg_reward is not None and len(ma)==ma.maxlen and np.mean(ma)>=solve_avg_reward:
            print(f"Solved with MA100={np.mean(ma):.2f} at episode {ep}")
            break

    return scores, agent

# ---- Unity adapter (if needed) ----
class UnityLikeWrapper:
    """Wrap a Unity env to a Gym-like API (adjust to your exact unity interface)."""
    def __init__(self, unity_env, train_mode=True):
        self.env = unity_env
        self.train_mode = train_mode
        # set brain/behavior here if required

    def reset(self):
        ts = self.env.reset(train_mode=self.train_mode)
        # extract vector state; return np.array
        return ts.vector_observations[0]

    def step(self, action):
        ts = self.env.step(action)
        s2 = ts.vector_observations[0]
        r  = ts.rewards[0]
        d  = ts.local_done[0]
        return s2, r, d, {}
