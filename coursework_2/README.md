# Udacity DRLND – Project 2: Continuous Control (CLI)

## How to run
```bash
# Train
python train.py --env_path path/to/Reacher_Windows_x86_64/Reacher.exe --episodes 2000

# Evaluate (renders)
python eval.py --env_path path/to/Reacher_Windows_x86_64/Reacher.exe --actor checkpoint_actor.pth --episodes 5
