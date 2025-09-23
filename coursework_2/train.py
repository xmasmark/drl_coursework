# coursework_2/train.py
"""
Train DDPG on Udacity's Reacher (single-agent) via CLI.
Usage:
    python train.py --env_path path/to/Reacher_Windows_x86_64/Reacher.exe
"""
import argparse
import csv
import time
from collections import deque
import numpy as np
import torch

from unityagents import UnityEnvironment
from ddpg_agent import Agent, DEVICE

def train(env_path, n_episodes=2000, max_t=1000, solve_avg=30.0, window=100,
          seed=2, save_prefix="checkpoint", log_csv="training_log.csv"):
    np.random.seed(seed); torch.manual_seed(seed)

    env = UnityEnvironment(file_name=env_path, seed=seed, no_graphics=True)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # get state/action sizes
    env_info = env.reset(train_mode=True)[brain_name]
    state = env_info.vector_observations[0]
    state_size = state.shape[0]
    action_size = brain.vector_action_space_size

    agent = Agent(state_size=state_size, action_size=action_size, random_seed=seed)

    scores = []
    scores_window = deque(maxlen=window)

    # CSV logger
    with open(log_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "score", f"avg_{window}"])

        for i_episode in range(1, n_episodes+1):
            env_info = env.reset(train_mode=True)[brain_name]
            state = env_info.vector_observations[0]
            agent.reset()
            score = 0.0
            t0 = time.time()

            for t in range(max_t):
                action = agent.act(state, add_noise=True)
                env_info = env.step(action)[brain_name]
                next_state = env_info.vector_observations[0]
                reward = env_info.rewards[0]
                done = env_info.local_done[0]

                agent.step(state, action, reward, next_state, done, t)
                state = next_state
                score += reward
                if done:
                    break

            scores.append(score)
            scores_window.append(score)
            avg = np.mean(scores_window)
            writer.writerow([i_episode, f"{score:.3f}", f"{avg:.3f}"])
            f.flush()
            dt = time.time()-t0
            print(f"Episode {i_episode:4d} | Score: {score:6.2f} | Avg({window}): {avg:6.2f} | {dt:.1f}s")

            # checkpoint occasionally
            if i_episode % 50 == 0:
                torch.save(agent.actor_local.state_dict(),  f"{save_prefix}_actor.pth")
                torch.save(agent.critic_local.state_dict(), f"{save_prefix}_critic.pth")

            # solved condition
            if avg >= solve_avg and i_episode >= window:
                print(f"Solved in {i_episode} episodes! Moving average >= {solve_avg}")
                torch.save(agent.actor_local.state_dict(),  f"{save_prefix}_actor.pth")
                torch.save(agent.critic_local.state_dict(), f"{save_prefix}_critic.pth")
                break

    env.close()
    return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_path", type=str, required=True,
                        help="Path to Unity Reacher executable (single-agent build).")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--max_t", type=int, default=1000)
    parser.add_argument("--solve_avg", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()

    train(env_path=args.env_path, n_episodes=args.episodes, max_t=args.max_t,
          solve_avg=args.solve_avg, seed=args.seed)
