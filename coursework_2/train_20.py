# coursework_2/train_20.py
"""
Train DDPG on Udacity's Reacher (20-agents) via CLI.

Usage:
    python train_20.py --env_path envs/Reacher_Windows_x86_64_20/Reacher.exe --episodes 1000
"""
import argparse
import csv
import time
from collections import deque
import numpy as np
import torch

from unityagents import UnityEnvironment
from ddpg_agent import Agent, DEVICE

def train_multi(env_path, n_episodes=1000, max_t=1000, solve_avg=30.0, window=100,
                seed=2, save_prefix="checkpoint_20", log_csv="training_log_20.csv"):
    np.random.seed(seed); torch.manual_seed(seed)

    env = UnityEnvironment(file_name=env_path, seed=seed, no_graphics=True)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True)[brain_name]
    states = env_info.vector_observations              # shape (num_agents, 33)
    num_agents = states.shape[0]
    state_size = states.shape[1]
    action_size = brain.vector_action_space_size

    # Minimal tweaks vs single-agent: learn earlier; one update per step
    agent = Agent(state_size=state_size, action_size=action_size,
                  random_seed=seed,
                  # keep your tuned values; just adjust warmup + updates_per_step
                  # actor_lr=5e-5, critic_lr=1e-3, batch_size=256, noise_sigma=0.15,
                  start_learn_after=1000,   # lots of data arrives quickly
                  updates_per_step=1)

    scores_history = []
    avg_history = deque(maxlen=window)

    with open(log_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "mean_score", f"avg_{window}", "sec"])

        for i_episode in range(1, n_episodes+1):
            t0 = time.time()
            env_info = env.reset(train_mode=True)[brain_name]
            states = env_info.vector_observations            # (N, state_size)
            agent.reset()
            scores = np.zeros(num_agents, dtype=np.float32)

            for t in range(max_t):
                # actions for all agents
                actions = np.vstack([agent.act(s, add_noise=True) for s in states])  # (N, action_size)
                env_info = env.step(actions)[brain_name]

                next_states = env_info.vector_observations   # (N, state_size)
                rewards     = np.array(env_info.rewards)     # (N,)
                dones       = np.array(env_info.local_done)  # (N,)

                # store all transitions
                for i in range(num_agents):
                    agent.step(states[i], actions[i], rewards[i], next_states[i], dones[i], t)

                scores += rewards
                states = next_states

                if np.any(dones):
                    break

            # Udacity metric: mean score across agents for this episode
            ep_mean = np.mean(scores).item()
            scores_history.append(ep_mean)
            avg_history.append(ep_mean)
            avg100 = float(np.mean(avg_history))
            dt = time.time() - t0

            writer.writerow([i_episode, f"{ep_mean:.3f}", f"{avg100:.3f}", f"{dt:.1f}"])
            f.flush()
            print(f"Episode {i_episode:4d} | MeanScore: {ep_mean:6.2f} | Avg({window}): {avg100:6.2f} | {dt:.1f}s")

            # periodic checkpoint
            if i_episode % 20 == 0:
                torch.save(agent.actor_local.state_dict(),  f"{save_prefix}_actor.pth")
                torch.save(agent.critic_local.state_dict(), f"{save_prefix}_critic.pth")

            # solved?
            if avg100 >= solve_avg and i_episode >= window:
                print(f"Solved in {i_episode} episodes! Moving average >= {solve_avg}")
                torch.save(agent.actor_local.state_dict(),  f"{save_prefix}_actor.pth")
                torch.save(agent.critic_local.state_dict(), f"{save_prefix}_critic.pth")
                break

    env.close()
    return scores_history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_path", type=str, required=True,
                        help="Path to Unity Reacher 20-agents executable.")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max_t", type=int, default=1000)
    parser.add_argument("--solve_avg", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()

    train_multi(env_path=args.env_path, n_episodes=args.episodes, max_t=args.max_t,
                solve_avg=args.solve_avg, seed=args.seed)
