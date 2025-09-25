# Collaboration and Competition (Tennis) – Report

**Algorithm:** MADDPG (centralized critics, decentralized actors)

## Environment
- 2 agents, continuous action space (2), state size 24 each.
- Reward +0.1 for returning ball; small negatives when letting it fall.

## Network Architectures
- Actor: [256, 256], Tanh head → [-1, 1].
- Critic: centralized input (concat all states + all actions), [256, 256].

## Key Hyperparameters
- γ=0.99, τ=1e-3, buffer=1e6, batch=256
- lr_actor=1e-3, lr_critic=1e-3, weight_decay=0
- OU(θ=0.15, σ=0.2)
- update_every=2, updates_per_step=2

## Results
- Solved when 100-episode average ≥ 0.5.
- Learning curve shown; final mean score: ...

## Ideas Tried / Stability Tricks
- Multiple updates per step boosted sample efficiency.
- Gradient clipping on critic (1.0).
- Soft updates τ=1e-3 stable; τ larger caused oscillations.

## Future Work
- Parameter sharing for actors, prioritized replay, TD3-style target policy smoothing, layer norm.
