import numpy as np
import torch
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment
from dqn_agent import Agent

# 1. Load Unity env (adjust filename if different)
# env = UnityEnvironment(file_name="Banana_Windows_x86_64/Banana.exe")
env = UnityEnvironment(file_name="Banana.exe")

# 2. Get default brain and sizes
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]

state_size = len(env_info.vector_observations[0])  # should be 37
action_size = brain.vector_action_space_size      # should be 4

agent = Agent(state_size=state_size, action_size=action_size, seed=0)

# 3. Training loop
def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    scores = []
    scores_window = []
    eps = eps_start
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
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
        eps = max(eps_end, eps_decay*eps)

        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window[-100:]):.2f}', end="")
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window[-100:]):.2f}')
        if np.mean(scores_window[-100:]) >= 13.0:
            print(f"\nEnvironment solved in {i_episode} episodes!\tAverage Score: {np.mean(scores_window[-100:]):.2f}")
            torch.save(agent.qnetwork_local.state_dict(), "checkpoint.pth")
            break
    return scores

scores = dqn()

# 4. Plot learning curve
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

env.close()
