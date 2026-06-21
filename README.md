# Space Gladiators — Orbit Wars

Competitive agents for the [Kaggle Orbit Wars](https://www.kaggle.com/competitions/orbit-wars) competition.

## Agents

| Agent | Dir | Description |
|---|---|---|
| **lite_exp50** | `sota_agents/lite_exp50/` | 6 waves mid-game, 9 terminal, 3 size tiers (50/75/100). Includes fleet-in-transit awareness for regroup decisions. |
| **lite_exp50_baseline** | `sota_agents/lite_exp50_baseline/` | Same as above without fleet-in-transit awareness (pre-improvement snapshot for A/B testing). |
| **strong_agent** | `sota_agents/strong_agent/` | 7 waves mid-game, 8 terminal, 2 size tiers (50/100). Supports learned scorer + REINFORCE training. |

## Requirements

- Python 3.10+
- `torch` (any CPU/CUDA)
- `kaggle_environments` (`pip install kaggle_environments`)

## Running a game

```bash
python visualise.py sota_agents/lite_exp50/main.py random
```

Arguments: `<agent1.py> <agent2.py> [seed] [output.json]`

- Agents can be Python files or built-in names (`random`, `greedy`, etc.)
- Runs a full 500-step game and saves `replay.json`
- Open `viewer.html` in a browser, click **Load Replay**, select `replay.json`

```bash
# 2P agents in the same file (seed 42, custom output)
python visualise.py sota_agents/lite_exp50/main.py \
  sota_agents/lite_exp50_baseline/main.py 42 my_replay.json
```

## Building for Kaggle

```bash
AGENT=sota_agents/lite_exp50
mkdir -p build
cp "$AGENT"/main.py build/
cp -r "$AGENT"/orbit_lite build/
tar -czf submission.tar.gz -C build .
rm -rf build
```

Upload to the [competition page](https://www.kaggle.com/competitions/orbit-wars/submissions) or use the CLI:

```bash
kaggle competitions submit orbit-wars -f submission.tar.gz -m "message"
```

## Config comparison

| Param | lite_exp50 | strong_agent |
|---|---|---|
| `max_waves_per_turn` (mid-game) | 6 | 7 |
| `terminal_max_waves_per_turn` | 9 | 8 |
| `size_multipliers` | (0.5, 0.75, 1.0) | (0.5, 1.0) |
| `reinforce_size_beta` | — | 2.2 |
| `enable_regroup` (mid-game) | True | True |
| `enable_regroup` (terminal) | False | False |

## strong_agent extras

### REINFORCE training

```bash
cd sota_agents/strong_agent
python train.py --episodes 2000 --lr 1e-3 --temperature 1.5
```

Trains a tiny MLP scorer via self-play against the hand-crafted baseline.

### Hyperparameter tuning

```bash
cd sota_agents/strong_agent
python hyper_sweep.py --games 30   # each variant vs fixed baseline
python hyper_grid.py --games 3     # round-robin tournament
```
