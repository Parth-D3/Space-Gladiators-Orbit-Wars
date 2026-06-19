#!/usr/bin/env python3
"""REINFORCE self-play training for the learned scoring network.

Trains ``ScoreNetwork`` by playing games against the baseline
(hand-crafted ``competitive_score``) and updating via policy gradient.

Usage
-----
    python train.py --episodes 2000 --lr 1e-3 --temperature 1.5
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import deque

import torch
import torch.optim as optim

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from learned_scorer import ScoreNetwork
from main import (
    ProducerLiteRuntime,
    single_obs_to_tensor,
    sparse_action_row_to_moves,
)


def _make_agent(runtime: ProducerLiteRuntime, *, no_grad: bool = True):
    """Wrap a runtime into a callable the kaggle environment expects.

    Set *no_grad=False* during REINFORCE training so gradients flow
    through the scoring network.
    """
    def agent(obs):
        player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
        player_id = int(player)
        obs_tensors = single_obs_to_tensor(obs, player_id=player_id)
        if no_grad:
            with torch.no_grad():
                sparse_row = runtime.tensor_action(obs_tensors)
        else:
            sparse_row = runtime.tensor_action(obs_tensors)
        return sparse_action_row_to_moves(sparse_row, obs, player_id=player_id)
    return agent


def run_episode(
    env,
    training_agent,
    baseline_agent,
    *,
    seed: int | None = None,
) -> tuple[float, list[torch.Tensor]]:
    """Play one game, return ``(final_reward, episode_log_probs)``.

    Reward is +1 for a win, -1 for a loss, 0 for a draw (ties happen in 4P).
    Log probs are the per-wave log-probabilities collected during the episode
    by the stochastic greedy selector.
    """
    env.run([training_agent, baseline_agent])
    steps = env.steps
    # Training agent is player index 0.
    final_reward = float(steps[-1][0]["reward"])
    return final_reward


def evaluate(model: ScoreNetwork, env, baseline_agent, *, n_games: int = 20) -> float:
    """Win-rate of *model* (deterministic, ``stochastic=False``) vs baseline."""
    wins = 0
    for game in range(n_games):
        runtime = ProducerLiteRuntime(scorer=model, stochastic=False)
        agent = _make_agent(runtime)
        env.run([agent, baseline_agent])
        reward = float(env.steps[-1][0]["reward"])
        if reward > 0:
            wins += 1
    return wins / max(n_games, 1)


def train(
    num_episodes: int = 1000,
    hidden: int = 32,
    lr: float = 1e-3,
    temperature: float = 1.0,
    save_every: int = 100,
    eval_every: int = 50,
    baseline_ema: float = 0.95,
    seed: int = 42,
) -> ScoreNetwork:
    """Run REINFORCE with a running-return baseline.

    Returns the trained model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ScoreNetwork(hidden=hidden).to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Load baseline agent (hand-crafted competitive_score, no scorer).
    from main import agent as baseline_agent_fn

    # Environment — non-debug for speed (no step recording), debug for reward.
    from kaggle_environments import make as kgl_make
    env = kgl_make("orbit_wars", configuration={"seed": seed}, debug=False)

    running_baseline = 0.0
    recent_returns: deque[float] = deque(maxlen=100)

    for ep in range(num_episodes):
        # --- create a fresh training runtime ---------------------------------
        runtime = ProducerLiteRuntime(
            scorer=model,
            stochastic=True,
            temperature=temperature,
        )
        training_agent = _make_agent(runtime, no_grad=False)
        baseline_agent = baseline_agent_fn  # re-use singleton

        # --- play one episode -----------------------------------------------
        env.reset()
        try:
            final_reward = run_episode(
                env,
                training_agent,
                baseline_agent,
                seed=seed + ep if seed else None,
            )
        except Exception as exc:
            print(f"  ⚠  episode {ep} failed: {exc}")
            continue

        recent_returns.append(final_reward)
        running_baseline = baseline_ema * running_baseline + (1 - baseline_ema) * final_reward

        # --- collect log probs from the runtime memory -----------------------
        log_probs = runtime.memory.episode_log_probs
        if not log_probs:
            continue

        # --- REINFORCE update ------------------------------------------------
        lp_tensor = torch.stack([lp.to(device) for lp in log_probs])
        advantage = final_reward - running_baseline
        loss = -lp_tensor.sum() * advantage

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # --- logging ---------------------------------------------------------
        if ep % max(save_every, 1) == 0 or ep == num_episodes - 1:
            mean_ret = float(sum(recent_returns)) / max(len(recent_returns), 1)
            win_rate = sum(1 for r in recent_returns if r > 0e0) / max(len(recent_returns), 1)
            print(
                f"ep={ep:5d}  reward={final_reward:+.1f}  "
                f"baseline={running_baseline:+.3f}  "
                f"loss={loss.item():+.4f}  "
                f"win100={win_rate:.3f}  "
                f"meanR={mean_ret:+.3f}  "
                f"actions={len(log_probs)}"
            )
            ckpt = os.path.join(_HERE, f"scorer_ep{ep}.pt")
            torch.save(model.state_dict(), ckpt)

        if ep % max(eval_every, 1) == 0:
            model.eval()
            from kaggle_environments import make as kgl_make
            eval_env = kgl_make(
                "orbit_wars",
                configuration={"seed": seed + 9999},
                debug=False,
            )
            wr = evaluate(model, eval_env, baseline_agent_fn, n_games=10)
            model.train()
            print(f"         eval_win_rate={wr:.3f}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="REINFORCE training for ScoreNetwork")
    parser.add_argument("--episodes", type=int, default=1000, help="number of training episodes")
    parser.add_argument("--hidden", type=int, default=32, help="MLP hidden size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--temperature", type=float, default=1.0, help="softmax temperature for exploration")
    parser.add_argument("--save-every", type=int, default=100, help="checkpoint interval")
    parser.add_argument("--eval-every", type=int, default=50, help="evaluation interval")
    parser.add_argument("--seed", type=int, default=42, help="base RNG seed")
    args = parser.parse_args()

    train(
        num_episodes=args.episodes,
        hidden=args.hidden,
        lr=args.lr,
        temperature=args.temperature,
        save_every=args.save_every,
        eval_every=args.eval_every,
        seed=args.seed,
    )
