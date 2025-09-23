# coursework_2/eval.py
"""
Evaluate a trained DDPG actor (deterministic, no noise).
Usage:
    python eval.py --env_path path/to/Reacher.exe --actor checkpoint_actor.pth --episodes 5
"""
import argparse
import numpy as np
import torch
from unityagents import UnityEnvironment
from model import Actor

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_actor(path, state_size, action_size):
    actor = Actor(state_size, action_size)
    actor.load_state_dict(torch.load(path, map_location=DEVICE))
    actor.to(DEVICE)
    actor.eval()
    return actor

def eval_agent(env_path, actor_path, episodes=5, max_t=1000, seed=2):
    env = UnityEnvironment(file_name=env_path, seed=seed, no_graphics=False)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=False)[brain_name]
    state_size = env_info.vector_observations[0].shape[0]
    action_size = brain.vector_action_space_size

    actor = load_actor(actor_path, state_size, action_size)

    def act_det(state):
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            a = actor(state_t).cpu().data.numpy().squeeze(0)
        return np.clip(a, -1, 1)

    scores = []
    for i in range(episodes):
        env_info = env.reset(train_mode=False)[brain_name]
        state = env_info.vector_observations[0]
        score = 0.0
        for t in range(max_t):
            action = act_det(state)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            score += reward
            state = next_state
            if done:
                break
        scores.append(score)
        print(f"[Eval] Episode {i+1}/{episodes} score: {score:.2f}")

    print(f"[Eval] Mean score over {episodes} episodes: {np.mean(scores):.2f}")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_path", type=str, required=True)
    parser.add_argument("--actor", type=str, default="checkpoint_actor.pth")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max_t", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()

    eval_agent(env_path=args.env_path, actor_path=args.actor,
               episodes=args.episodes, max_t=args.max_t, seed=args.seed)
