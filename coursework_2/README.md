# Udacity DRLND – Project 2: Continuous Control (CLI)

This project solves the **Continuous Control** task from the Udacity Deep Reinforcement Learning Nanodegree.  
The agent controls 20 double-jointed arms to keep their end-effectors in the target location.  

The environment is considered solved when the **average score over 100 consecutive episodes** reaches **≥30** across all agents.

---

## Environment
- Unity Reacher (20 agents version)
- State space: 33 dimensions (positions, rotations, velocities, etc.)
- Action space: 4 continuous actions ∈ [-1, 1]

---

## Installation
Clone this repo and install requirements:
```bash
pip install -r requirements.txt



## How to run
```bash
# Train
python train_20.py --env_path "envs/Reacher_Windows_x86_64_20/Reacher.exe" --episodes 200 --max_t 1000 --seed 0 --worker_id 1

