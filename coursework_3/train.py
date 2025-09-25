import numpy as np
import torch
from collections import deque
from maddpg_agent import MADDPG
# from unityagents import UnityEnvironment  # ensure installed in your env

# ENV_PATH = "Tennis_Windows_x86_64/Tennis.exe"  # adjust for your OS
ENV_PATH = "envs/Tennis_Windows_x86_64/Tennis.exe"


def train(n_episodes=3000, max_t=1000, print_every=100, target_score=0.5):
    env = UnityEnvironment(file_name=ENV_PATH, seed=1, no_graphics=True)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True)[brain_name]
    n_agents = len(env_info.agents)
    state_size = env_info.vector_observations.shape[1]
    action_size = brain.vector_action_space_size

    maddpg = MADDPG(state_size, action_size, n_agents=n_agents)

    scores_window = deque(maxlen=100)
    all_scores = []

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        maddpg.reset()
        scores = np.zeros(n_agents)

        for t in range(max_t):
            actions = maddpg.act(states, add_noise=True)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            maddpg.step(states, actions, rewards, next_states, dones)

            states = next_states
            scores += rewards
            if np.any(dones):
                break

        episode_score = np.max(scores)  # Udacity metric uses max agent score per episode
        scores_window.append(episode_score)
        all_scores.append(episode_score)

        if i_episode % print_every == 0:
            print(f"Episode {i_episode}\tAverage Score: {np.mean(scores_window):.3f}")

        if np.mean(scores_window) >= target_score:
            print(f"Solved in {i_episode} episodes! Average Score: {np.mean(scores_window):.3f}")
            # save weights
            for i, ag in enumerate(maddpg.agents):
                torch.save(ag.actor_local.state_dict(), f"checkpoints/actor_local_{i}.pth")
                torch.save(ag.critic_local.state_dict(), f"checkpoints/critic_local_{i}.pth")
            break

    env.close()
    return all_scores

if __name__ == "__main__":
    train()
