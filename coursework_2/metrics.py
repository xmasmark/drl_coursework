import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("training_log_20.csv")
df['episode'] = df['episode'].astype(int)
df['mean_score'] = df['mean_score'].astype(float)
df['avg_100'] = df['avg_100'].astype(float)

plt.figure(figsize=(10,5))
plt.plot(df['episode'], df['mean_score'], label="Mean Score (per episode)", alpha=0.6)
plt.plot(df['episode'], df['avg_100'], label="Moving Avg (100 episodes)", linewidth=2)
plt.axhline(30, color="r", linestyle="--", label="Solve Threshold (30)")
plt.xlabel("Episode")
plt.ylabel("Score")
plt.title("DDPG Agent Training on 20-Agent Reacher Environment")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_curve.png")
plt.show()
