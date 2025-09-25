
# train.py
import os
import numpy as np
import torch
from collections import deque
from unityagents import UnityEnvironment
from maddpg_agent import MADDPG

# ---- Adjust if your path differs ----
ENV_PATH = "envs/Tennis_Windows_x86_64/Tennis.exe"

# Reproducibility
SEED = 1
np.random.seed(SEED)
torch.manual_seed(SEED)

WARMUP_EPISODES = 200

def train(n_episodes=3000, max_t=1000, print_every=50, target_score=0.5):
    # Make sure checkpoints/ exists
    os.makedirs("checkpoints", exist_ok=True)

    # Boot env
    env = UnityEnvironment(file_name=ENV_PATH, seed=SEED, no_graphics=True)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # Inspect spaces
    env_info = env.reset(train_mode=True)[brain_name]
    n_agents = len(env_info.agents)
    state_size = env_info.vector_observations.shape[1]
    action_size = brain.vector_action_space_size
    print(f"[Init] Agents: {n_agents}, state_size: {state_size}, action_size: {action_size}")

    # Build agent
    maddpg = MADDPG(state_size, action_size, n_agents=n_agents, seed=SEED)

    scores_window = deque(maxlen=100)  # rolling avg over last 100 episodes
    all_scores = []

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations  # shape (n_agents, state_size)

        # OU noise sigma decay: 0.20 -> 0.05 over first 500 episodes
        if hasattr(maddpg, "set_noise_sigma"):
            frac = min(1.0, (i_episode - 1) / 1000.0)
            sigma = 0.30 * (1.0 - frac) + 0.05 * frac
            maddpg.set_noise_sigma(sigma)

        maddpg.reset()
        scores = np.zeros(n_agents)

        for t in range(max_t):

            actions = maddpg.act(states, add_noise=True).astype(np.float32)  # (n_agents, action_size) in [-1,1]
            if i_episode <= WARMUP_EPISODES:
                actions = np.random.uniform(-1, 1, size=(n_agents, action_size)).astype(np.float32)
            else:
                actions = maddpg.act(states, add_noise=True).astype(np.float32)

            # env_info = env.step(actions)[brain_name]

            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards               # list len n_agents
            dones = env_info.local_done              # list len n_agents

            maddpg.step(states, actions, rewards, next_states, dones)

            states = next_states
            scores += rewards
            if np.any(dones):
                break

        # Udacity metric uses max agent score per episode
        episode_score = np.max(scores)
        scores_window.append(episode_score)
        all_scores.append(episode_score)

        if i_episode % print_every == 0:
            print(f"Episode {i_episode:4d}\tAvg(100): {np.mean(scores_window):.3f}\tLast: {episode_score:.3f}")

        if np.mean(scores_window) >= target_score:
            avg = np.mean(scores_window)
            print(f"\n🎉 Solved in {i_episode} episodes! 100-episode average: {avg:.3f}\nSaving checkpoints...")
            for i, ag in enumerate(maddpg.agents):
                torch.save(ag.actor_local.state_dict(),  f"checkpoints/actor_local_{i}.pth")
                torch.save(ag.critic_local.state_dict(), f"checkpoints/critic_local_{i}.pth")
            break

    env.close()
    return all_scores


if __name__ == "__main__":
    train()



# import numpy as np
# import torch
# from collections import deque
# from maddpg_agent import MADDPG
# from unityagents import UnityEnvironment

# # from unityagents import UnityEnvironment  # ensure installed in your env

# # ENV_PATH = "Tennis_Windows_x86_64/Tennis.exe"  # adjust for your OS
# ENV_PATH = "envs/Tennis_Windows_x86_64/Tennis.exe"


# def train(n_episodes=3000, max_t=1000, print_every=100, target_score=0.5):
#     env = UnityEnvironment(file_name=ENV_PATH, seed=1, no_graphics=True)
#     brain_name = env.brain_names[0]
#     brain = env.brains[brain_name]

#     env_info = env.reset(train_mode=True)[brain_name]
#     n_agents = len(env_info.agents)
#     state_size = env_info.vector_observations.shape[1]
#     action_size = brain.vector_action_space_size

#     maddpg = MADDPG(state_size, action_size, n_agents=n_agents)

#     scores_window = deque(maxlen=100)
#     all_scores = []

#     for i_episode in range(1, n_episodes+1):
#         env_info = env.reset(train_mode=True)[brain_name]
#         states = env_info.vector_observations
#         maddpg.reset()
#         scores = np.zeros(n_agents)

#         for t in range(max_t):
#             actions = maddpg.act(states, add_noise=True)
#             env_info = env.step(actions)[brain_name]
#             next_states = env_info.vector_observations
#             rewards = env_info.rewards
#             dones = env_info.local_done

#             maddpg.step(states, actions, rewards, next_states, dones)

#             states = next_states
#             scores += rewards
#             if np.any(dones):
#                 break

#         episode_score = np.max(scores)  # Udacity metric uses max agent score per episode
#         scores_window.append(episode_score)
#         all_scores.append(episode_score)

#         if i_episode % print_every == 0:
#             print(f"Episode {i_episode}\tAverage Score: {np.mean(scores_window):.3f}")

#         if np.mean(scores_window) >= target_score:
#             print(f"Solved in {i_episode} episodes! Average Score: {np.mean(scores_window):.3f}")
#             # save weights
#             for i, ag in enumerate(maddpg.agents):
#                 torch.save(ag.actor_local.state_dict(), f"checkpoints/actor_local_{i}.pth")
#                 torch.save(ag.critic_local.state_dict(), f"checkpoints/critic_local_{i}.pth")
#             break

#     env.close()
#     return all_scores

# if __name__ == "__main__":
#     train()
