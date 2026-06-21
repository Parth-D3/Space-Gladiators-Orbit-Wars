# Training the Strong Agent

## Quick Start

```bash
cd sota_agents/strong_agent
python train_ppo.py --episodes 2000 --batch-size 8
```

## CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--episodes` | 500 | Total training episodes |
| `--batch-size` | 4 | Episodes per PPO update |
| `--k-epochs` | 3 | PPO epochs per batch |
| `--lr` | 1e-3 | Learning rate |
| `--temperature` | 1.5 | Softmax exploration temperature |
| `--hidden` | 32 | MLP hidden size |
| `--strategic-scale` | 0.01 | Dense reward weight (0 = standard win/loss only) |
| `--save-every` | 50 | Checkpoint interval |
| `--eval-every` | 50 | Evaluation interval |
| `--seed` | 42 | RNG seed |

## Log Files

Created automatically in `logs/` when training starts. Three files per run:

| File | Frequency | Contents |
|---|---|---|
| `ppo_train_<timestamp>.csv` | Per update | winrate, mean_return, strategic rewards, policy/value loss, entropy, gradient norm, explained variance, wins/losses/draws, SPS |
| `ppo_episodes_<timestamp>.csv` | Per game | final_reward, strategic_reward, combined_reward, opponent, waves |
| `ppo_eval_<timestamp>.csv` | Per eval | eval win rate over 10 held-out games |

### REINFORCE alternative

`train.py` uses the same infrastructure — creates `reinforce_*.csv` logs.

## Weights

Checkpoints saved to the agent directory:

- `scorer_ppo_ep{episode}.pt` — policy weights every `--save-every` episodes
- `value_ppo_ep{episode}.pt` — value network weights
- `scorer_ppo_best.pt` / `value_ppo_best.pt` — best eval win rate

## What to Send Back

```
logs/ppo_train_*.csv
logs/ppo_episodes_*.csv
logs/ppo_eval_*.csv
scorer_ppo_best.pt
value_ppo_best.pt
```

Optionally include the full run: all `logs/*` + intermediate checkpoints.
