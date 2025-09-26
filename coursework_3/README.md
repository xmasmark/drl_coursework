# Collaboration and Competition (Tennis) — MADDPG

Solve the Unity **Tennis** environment with **MADDPG** (centralized critics, decentralized actors). Two rackets cooperate to keep the ball in play. The project is considered *solved* when the 100-episode moving average of the per-episode **max agent** score is **≥ 0.5**.

---

## Environment

- **Agents:** 2  
- **State:** 24 dims per agent (8 obs × 3 stacked)  
- **Action:** 2-dim continuous in \[-1, 1] per agent  
- **Reward:** +0.1 for successful return; small negatives for letting the ball hit the ground or going out  
- **Success criterion:** average score ≥ 0.5 over 100 episodes (Udacity)

---

## Algorithm

**MADDPG**: each agent has its own **actor**; each agent’s **critic** is **centralized** (sees all states + all actions).

Key ingredients:
- Shared replay buffer with **per-agent rewards & dones** (correct credit assignment)
- OU noise for exploration (σ decays over episodes)
- Multiple updates per environment step
- Soft target updates (τ)

---

## Project Structure

coursework_3/
├─ envs/
│ └─ Tennis_Windows_X86_64/
│ └─ Tennis.exe
├─ model.py # Actor & centralized Critic networks
├─ maddpg_agent.py # Agents + MADDPG logic (per-agent r/done fix)
├─ replay_buffer.py # Shared buffer (rewards/dones shaped (B, n_agents))
├─ noise.py # OU noise
├─ train.py # Training loop (warmup, sigma decay, saves on success)
├─ eval.py # Deterministic evaluation & plot (optional)
├─ Report.md # (to be filled)
└─ checkpoints/ # Saved weights (created on success)

## How to Run

cd coursework_3
python train.py

Logs print every 50 episodes.

Saves checkpoints only when the success criterion is met:

checkpoints/actor_local_0.pth, actor_local_1.pth

checkpoints/critic_local_0.pth, critic_local_1.pth

## Results

Solved in 995 episodes

100-episode average at solve: 0.501

Checkpoints saved to checkpoints/.

## Hyperparameters (final run)
Parameter	Value
Discount γ	0.99
Soft update τ	1e-3
Replay buffer	1e6
Batch size	256
Actor LR	1e-4
Critic LR	1e-3
Critic L2 (weight decay)	1e-5
Updates per step	4
Random warmup	200 episodes (uniform [-1,1] actions)
OU σ decay	0.30 → 0.05 (first 1000 episodes)

## Networks

Actor: FC(256) → FC(256) → Tanh head (actions in [-1,1])

Critic (centralized): FC(256) on full state, concat full action → FC(256) → Q

## Implementation Notes

Per-agent credit assignment: each critic learns from its own reward_i and done_i (avoid averaging rewards across agents).

Action dtype: ensure actions passed to Unity are np.float32.

Exploration: OU noise with σ decay + early random warmup to seed the buffer with meaningful rallies.

Checkpoints: saved only on success to keep the folder clean.

## Troubleshooting

NameError: UnityEnvironment: add from unityagents import UnityEnvironment at the top of train.py.

ModuleNotFoundError: unityagents: pip install unityagents.

All zeros for many episodes: ensure env.step(actions) receives actions.astype(np.float32); keep warmup enabled.

Empty checkpoints/: not solved yet (by design, only saved on success).

## Acknowledgements

Udacity Deep Reinforcement Learning Nanodegree

Unity ML-Agents Tennis environment