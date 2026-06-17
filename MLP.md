# Orbit Wars: PPO Agent Architecture Specification

## 1. Objective and System Constraints

**Goal:** Implement a reinforcement learning agent for the "Orbit Wars" environment using Proximal Policy Optimization (PPO) in PyTorch.
**Constraints:**

* The environment expects a Python file (`main.py`) that returns an action array within a 1-second timeout.
* The submitted agent cannot depend on PyTorch. The training script must export the trained policy weights to a `.npz` file.
* A separate, pure-Numpy inference script must load these weights and execute the exact same forward pass using greedy argmax selection.

## 2. Observation Space (Inputs)

The neural network requires a fixed-size global context and a variable-size list of planet features. The raw observation must be pre-processed into two distinct tensors.

### A. Global Features Tensor (Shape: `[13]`)

* `player_id`: One-hot encoded.
* `angular_velocity`: Raw scalar from the environment.
* `global_stats`: Aggregated game statistics (e.g., total ships owned vs. total enemy ships, total planet production owned vs. enemy).

### B. Planet Features Tensor (Shape: `[N, 22]`)

For every planet $i \in N$, extract and calculate the following 22 features:

* **Identity & Capacity:**
* `is_mine`, `is_enemy`, `is_neutral` (One-hot encoded).
* `radius` (Defines ship production rate).


* **Ship Count (Anti-Freeze Scaling):**
* `ships_linear`: Current garrison, linearly scaled and clipped at $500$.
* `ships_log`: $\log(1 + \text{garrison})$.


* **Motion (Vectorized):**
* `v_x`, `v_y`: Tangential velocity for orbiting inner planets, path delta for comets, $0$ for static outer planets.


* **Engineered Threat/Defense Heuristics:**
Iterate over all active fleets in the environment. For each fleet whose angle aligns within $\approx 18^\circ$ of the vector from its source planet to planet $i$:
* `eta`: $\frac{\text{Euclidean Distance}}{\text{Fleet Speed}}$.
* `incoming_hostile`: Sum of ships from enemy fleets heading toward this planet.
* `incoming_allied`: Sum of ships from allied fleets heading toward this planet.
* `nearest_threat_eta`: The minimum `eta` of any hostile fleet (default $50.0$ if none).
* `farthest_threat_eta`: The maximum `eta` of any hostile fleet (default $0.0$ if none).
* `net_projected_garrison`: $\tanh((\text{Current Ships} + \text{incoming\_allied} - \text{incoming\_hostile}) / 200)$.



## 3. Network Architecture (Deep Sets MLP)

The network must be depth-agnostic to allow seamless translation to Numpy. Do not use Batch Normalization, as it complicates the Numpy export.

1. **Planet Encoder:**
* **Input:** `[N, 22]`
* **Structure:** 4-layer MLP, hidden width of 96, ReLU activations.
* **Output:** `[N, 96]` (Hidden representation per planet).


2. **Global Pooling:**
* Apply $\max$ pooling across the $N$ dimension $\rightarrow$ `[96]`.
* Apply $\text{mean}$ pooling across the $N$ dimension $\rightarrow$ `[96]`.
* Concatenate max, mean, and the Global Features Tensor (13 dims) $\rightarrow$ `[205]`.


3. **Context Block:**
* **Input:** `[205]`
* **Structure:** 4-layer MLP, hidden width of 160, ReLU activations.
* **Output:** `[160]` (Global game context).


4. **Multi-Head Outputs:**
Combine the global context `[160]` with the individual planet encodings `[N, 96]` as needed to generate:
* `source_logits` (`[N]`): Probability of launching from planet $i$.
* `target_logits` (`[N]`): Probability of targeting planet $i$.
* `fraction_logits` (`[5]`): Categorical logits for ship fraction $\{0, 0.25, 0.5, 0.75, \text{all\_but\_one}\}$.
* `value_estimate` (`[1]`): State value $V(s)$ for PPO advantage calculation.



## 4. Action Masking & Decoding

The network outputs logical intents. The environment requires physical commands formatted as `[from_planet_id, angle_in_radians, num_ships]`. Implement a deterministic decoder with the following rules:

* **Masking:**
* Mask `source_logits` to $-\infty$ for planets not owned by the agent, or owned planets with $< 1$ ship.
* Mask `target_logits` to $-\infty$ for planets that do not exist.


* **Intercept Solver (`utils.py`):** When a source and target are selected, calculate the intercept angle based on where the target *will be* upon arrival, not its current position. Uses a time-scan (up to 300 turns) to find the first feasible arrival, then bisects within that turn (20 iterations) for sub-turn precision on orbiting targets. Comet positions are looked up from their observed path data rather than re-simulated. Account for exact spawn offsets (planet center + radius + 0.1).
* **Lane Clearance (Sun & Planets):** Calculate the planned flight path. If it intersects the central sun or collides with an intermediate planet (segment-distance for static planets, half-turn sampling for moving planets), flag the lane as blocked.
* **Action Repair:** If the argmax `Source x Target` lane is blocked, do not default to a no-op. Walk down the sorted `target_logits` array until a safe, unobstructed lane is found.
* **Fraction Translation:** Convert the chosen fraction category into an absolute integer. If `all_but_one` is selected, leave exactly 1 ship behind to prevent self-elimination.

## 5. Training & Reward Formulation

* **Algorithm:** PPO with Generalized Advantage Estimation (GAE), entropy bonus, and gradient clipping.
* **Base Reward (Per-Step):** $\frac{\Delta(\text{my\_total\_ships} - \text{best\_opponent\_total\_ships})}{200}$.
* **Event Bonuses (one-shot, detected between steps):**
  * Planet captured: $+0.25 + 0.05 \times \text{production}$.
  * Planet lost (non-comet): $-0.25 - 0.05 \times \text{production} - \text{garrison}/100$.
  * Comet garrison expires: $-\text{ships\_lost}/50$.
  * Production advantage: $+0.001 \times \Delta(\text{my\_prod} - \text{opp\_prod})$ per step.
* **Comet Overcommit Penalty (during rollout):** Sending ships to a comet with fewer than 5 turns remaining costs $-\frac{n \times \max(0, 1 - t/5)}{200}$.
* **Sparse Reward (Terminal):** $+3$ for a win, $-3$ for a loss, $0$ for a draw.

### Monitoring Metrics (rolling 50-episode window)

| Metric | Label | Meaning |
|---|---|---|
| `cap` | Planets captured/ep | Rolling mean of enemy/neutral planets seized per game |
| `lost` | Planets lost/ep | Rolling mean of non-comet planets lost per game |
| `lane` | Lane-blocked rate | Fraction of policy-chosen source/target pairs that failed sun, obstacle, or intercept checks — high values indicate the policy is attempting invalid launches |

### Weights & Biases Logging

| Flag | Description |
|---|---|
| `--wandb` | Enable wandb logging |
| `--wandb-project` | Project name (default: `space-gladiators-orbit-wars`) |
| `--wandb-entity` | Wandb entity/username |
| `--wandb-tags` | Comma-separated tags for the run |

Metrics logged per update: `winrate`, `mean_ep_return`, `planets_captured`, `planets_lost`, `lane_blocked_rate`, `policy_loss`, `value_loss`, `entropy`, `steps_per_sec`, `wins`, `draws`, `losses`. Hyperparameters (learning rate, rollout length, etc.) are saved as run config. Gradients are tracked via `wandb.watch`. The model checkpoint is logged as an artifact at the end of training.

### Opponent Strategy

| `--opponent` | Description |
|---|---|
| `sniper` (default) | Nearest-planet heuristic (`sniper.py`) |
| `selfplay` | Greedy argmax copy of the current policy |
| `strong` | Torch-based strong agent (`sota_agents/strong_agent`) |
| `file:<path>` | Any custom agent file |

Randomize player seat assignments every episode. Mix of 1v1 and 4-player configurations via `--players mix`.