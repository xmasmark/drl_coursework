# plot_scores.py
import csv, matplotlib.pyplot as plt

episodes, ep, avg = [], [], []
with open("scores.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        episodes.append(int(row["episode"]))
        ep.append(float(row["episode_score"]))
        avg.append(float(row["avg100"]))

plt.figure(figsize=(8,4.5))
plt.plot(episodes, ep, label="Episode score")
plt.plot(episodes, avg, label="100-episode average")
plt.xlabel("Episode"); plt.ylabel("Score (max over 2 agents)")
plt.title("Tennis – Training Performance")
plt.legend(); plt.tight_layout(); plt.savefig("training_curve.png", dpi=150)
print("Saved training_curve.png")
