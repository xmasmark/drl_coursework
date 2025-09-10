import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque

from unityagents import UnityEnvironment
from dqn_agent import Agent

# ---- Reproducibility ----
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---- Load Unity env (keep the full build folder!) ----
exe = Path(__file__).with_name("Banana_Windows_x86_64") / "Banana.exe"  # adjust if needed
env = UnityEnvironment(file_name=str(exe), no_graphics=True, worker_id=1, base_port=5005)

# ---- Inspect brain & sizes ----
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]

state_size = len(env_info.vector_observations[0])   # 37
action_size = brain.vector_action_space_size        # 4

# Agent (supports both state_size/action_size or state_dim/action_dim)
agent = Agent(state_dim=state_size, action_dim=action_size, seed=SEED)

def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start

    try:
        for i_episode in range(1, n_episodes + 1):
            env_info = env.reset(train_mode=True)[brain_name]
            state = env_info.vector_observations[0]
            score = 0.0

            for t in range(max_t):
                action = agent.act(state, eps)
                env_info = env.step(action)[brain_name]
                next_state = env_info.vector_observations[0]
                reward = env_info.rewards[0]
                done = env_info.local_done[0]

                agent.step(state, action, reward, next_state, done)

                state = next_state
                score += reward
                if done:
                    break

            scores.append(score)
            scores_window.append(score)
            eps = max(eps_end, eps_decay * eps)

            avg100 = float(np.mean(scores_window))
            print(f"\rEpisode {i_episode:4d} | Avg100={avg100:6.2f} | eps={eps:5.3f}", end="")
            if i_episode % 100 == 0:
                print(f"\rEpisode {i_episode:4d} | Avg100={avg100:6.2f} | eps={eps:5.3f}")

            if len(scores_window) == 100 and avg100 >= 13.0:
                print(f"\nEnvironment solved in {i_episode} episodes! Avg100={avg100:.2f}")
                torch.save(agent.q.state_dict(), "checkpoint.pth")
                break

    finally:
        # Save outputs even if interrupted
        np.save("scores.npy", np.array(scores))
        plt.figure()
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel("Score")
        plt.xlabel("Episode #")
        plt.title("DQN on Banana Environment")
        plt.tight_layout()
        plt.savefig("scores.png", dpi=150)
        # (optional) comment out next line if you don't want a popup:
        # plt.show()

    return scores

scores = dqn()

# Clean shutdown
env.close()
print("\nSaved: checkpoint.pth (on solve), scores.npy, scores.png")
