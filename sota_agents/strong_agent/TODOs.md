# Improvement Ideas for strong_agent

## 1. Hyperparameter Optimization
Systematic search (Bayesian/random) over `ProducerLiteConfig` knobs: horizon, source/target caps, ROI threshold, regroup params, terminal phase timing, size multipliers. No code changes needed — just sweep and evaluate.

## 2. Greedy Selection → Beam/Rollout Search
`_greedy_select` picks one wave at a time with no lookahead. Use the fast garrison projector to do shallow rollouts over top-K candidate combinations.

## 3. Scoring Function Improvements
- Per-opponent weights instead of equal-weight sum (e.g., punish the leader in 4P)
- Non-linear scoring: `my_delta - softplus(sum(opponent_delta))`
- Add future production value of captured/defended planets
- Learned scoring network

## 4. Multi-Turn Coordination
Commit to multi-turn attack sequences (e.g., soft-enumeration over 3 turns) rather than re-evaluating from scratch each step.

## 5. Opponent Modeling
In 4P, infer opponent intent (which planets they're likely attacking) to improve defensive target selection and reinforcement margin.

## 6. Comet Handling
Comets are currently excluded from attack targets. They're high-value (free capture, high production) — incorporate as high-priority targets.

## 7. Fleet-in-Transit Awareness
The garrison projection tracks only planet-bound ships. Consider the value of ships currently flying for better resource allocation.

## 8. Learned Components
Replace hand-crafted scoring with a small learned network. Train via behavior cloning from a stronger agent or RL self-play. Keep geometry/movement/garrison pipeline fixed.

## 9. MCTS / True Search
Use the fast forward model to run Monte Carlo Tree Search over action sequences. The deterministic physics and fast garrison projector make this feasible.
