# eval.py
import os, numpy as np, torch
import matplotlib.pyplot as plt
from collections import deque
from unityagents import UnityEnvironment
from maddpg_agent import MADDPG

ENV_PATH = "envs/Tennis_Windows_x86_64/Tennis.exe"
SEED = 1

def load_trained(maddpg, ckpt_dir="checkpoints"):
    for i, ag in enumerate(maddpg.agents):
        ag.actor_local.load_state_dict(torch.load(os.path.join(ckpt_dir, f"actor_local_{i}.pth"), map_location="cpu"))
        ag.critic_local.load_state_dict(torch.load(os.path.join(ckpt_dir, f"critic_local_{i}.pth"), map_location="cpu"))
        # sync targets to locals
        ag.hard_update(ag.actor_target, ag.actor_local)
        ag.hard_update(ag.critic_target, ag.critic_local)

def evaluate(n_episodes=100, max_t=1000, plot_path="scores_eval.png"):
    env = UnityEnvironment(file_name=ENV_PATH, seed=SEED, no_graphics=True)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True)[brain_name]
    n_agents = len(env_info.agents)
    state_size = env_info.vector_observations.shape[1]
    action_size = brain.vector_action_space_size

    maddpg = MADDPG(state_size, action_size, n_agents=n_agents, seed=SEED)
    load_trained(maddpg)

    scores = []
    for i in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        ep_scores = np.zeros(n_agents)
        while True:
            actions = maddpg.act(states, add_noise=False).astype(np.float32)
            env_info = env.step(actions)[brain_name]
            states = env_info.vector_observations
            ep_scores += env_info.rewards
            if np.any(env_info.local_done):
                break
        scores.append(np.max(ep_scores))
        if i % 10 == 0:
            print(f"Eval episode {i:3d} | avg so far: {np.mean(scores):.3f}")

    env.close()

    # Plot
    ma = np.convolve(scores, np.ones(10)/10, mode="valid")  # 10-ep moving avg
    plt.figure(figsize=(8,4.5))
    plt.plot(scores, label="Eval score per episode")
    plt.plot(np.arange(len(ma)) + 9, ma, label="10-episode MA")
    plt.xlabel("Episode")
    plt.ylabel("Score (max over 2 agents)")
    plt.title("Tennis – Evaluation Performance (deterministic, noise OFF)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"Evaluation complete. Mean over {n_episodes} eps: {np.mean(scores):.3f}")
    print(f"Saved plot to {plot_path}")

if __name__ == "__main__":
    evaluate()
