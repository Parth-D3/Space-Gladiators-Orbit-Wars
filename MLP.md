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

### B. Planet Features Tensor (Shape: `[N, 21]`)

For every planet $i \in N$, extract and calculate the following 21 features:

* **Identity & Capacity:**
* `is_mine`, `is_enemy`, `is_neutral` (One-hot encoded).
* `radius` (Defines ship production rate).


* **Ship Count (Anti-Freeze Scaling):**
* `ships_linear`: Current garrison, strictly clipped at $500$.
* `ships_log`: $\log(\text{current\_garrison} - 500)$ if garrison $> 500$, else $0$.


* **Motion (Vectorized):**
* `v_x`, `v_y`: Tangential velocity for orbiting inner planets, path delta for comets, $0$ for static outer planets.


* **Engineered Threat/Defense Heuristics:**
Iterate over all active fleets in the environment. For each fleet targeting planet $i$:
* `eta`: $\frac{\text{Euclidean Distance}}{\text{Fleet Speed}}$. (Note: Fleet speed scales with fleet size, up to 6 units/turn).
* `incoming_hostile`: Sum of ships from enemy fleets targeting this planet.
* `incoming_allied`: Sum of ships from allied fleets targeting this planet.
* `nearest_threat_eta`: The minimum `eta` of any hostile fleet (default to a high constant if none).
* `net_projected_garrison`: $\text{Current Ships} + \text{incoming\_allied} - \text{incoming\_hostile}$.



## 3. Network Architecture (Deep Sets MLP)

The network must be depth-agnostic to allow seamless translation to Numpy. Do not use Batch Normalization, as it complicates the Numpy export.

1. **Planet Encoder:**
* **Input:** `[N, 21]`
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


* **Intercept Solver:** When a source and target are selected, calculate the intercept angle based on where the target *will be* upon arrival, not its current position. Account for exact spawn offsets (planet center + radius + 0.1).
* **Lane Clearance (Sun & Planets):** Calculate the planned flight path. If it intersects the central sun, crosses exactly 30 units past a moving target (near-miss check), or collides with an intermediate planet, flag the lane as blocked.
* **Action Repair:** If the argmax `Source x Target` lane is blocked, do not default to a no-op. Walk down the sorted `target_logits` array until a safe, unobstructed lane is found.
* **Fraction Translation:** Convert the chosen fraction category into an absolute integer. If `all_but_one` is selected, leave exactly 1 ship behind to prevent self-elimination.

## 5. Training & Reward Formulation

* **Algorithm:** PPO with Generalized Advantage Estimation (GAE), entropy bonus, and gradient clipping.
* **Dense Reward (Per-Step):** $\frac{\Delta(\text{my\_total\_ships} - \text{best\_opponent\_total\_ships})}{200}$.
* **Sparse Reward (Terminal):** $+3$ for a win, $-3$ for a loss, $0$ for a draw.
* **Opponent Strategy:** Randomize player seat assignments every episode. Train against a heuristic baseline (e.g., nearest-planet-sniper) using a mix of 1v1 and 4-player configurations.