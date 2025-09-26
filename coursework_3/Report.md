# Collaboration and Competition (Tennis) — Report (Markdown Stub)

**Note:** Please refer to the included **PDF** (`Report.pdf`) for the fully formatted write-up with equations, tables, and the evaluation plot.

## Summary
- **Algorithm:** MADDPG (centralized critics, decentralized actors)
- **Environment:** Unity Tennis (2 agents, continuous actions)
- **Result:** *Solved in 995 episodes* — 100-episode moving average **0.501**
- **Checkpoints:** Saved in `checkpoints/` upon reaching the success criterion

## Files
- `Report.pdf` — Full report with proper scientific formatting
- `scores_eval.png` — Deterministic evaluation over 100 episodes (noise OFF)
- `train.py`, `maddpg_agent.py`, `model.py`, `replay_buffer.py`, `noise.py`, `eval.py`

## Reproduce
```bash
python train.py          # train until success (saves checkpoints on solve)
python eval.py           # loads checkpoints, generates scores_eval.png
