# Project 1: Navigation (Udacity Deep Reinforcement Learning Nanodegree)

This project trains a Deep Q-Network (DQN) agent to navigate a large, square world and collect yellow bananas while avoiding blue bananas.  
The agent interacts with the [Unity ML-Agents Banana environment](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation).

---

## Project Details

- **State space:** 37-dimensional continuous vector, describing the agent’s velocity and a ray-based perception of objects in the environment.  
- **Action space:** 4 discrete actions:
  - `0`: move forward  
  - `1`: move backward  
  - `2`: turn left  
  - `3`: turn right  
- **Reward function:** +1 for collecting a yellow banana, -1 for collecting a blue banana.  
- **Goal:** achieve an average score of **+13 over 100 consecutive episodes**.  

My implementation solved the environment in **673 episodes**, significantly faster than the 1800 episode threshold.

---

## Getting Started

### Clone this repository
```bash
git clone https://github.com/xmasmark/drl_coursework.git
cd drl_coursework/coursework_1

### 1. Clone this repository
## Download the Banana environment and place it in the folder called Banana_Windows_x86_64

python -m venv .venv
.\.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # Linux/Mac

pip install --no-deps -r requirements.txt

#Finally run the code:

python train_banana.py

