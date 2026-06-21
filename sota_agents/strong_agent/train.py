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
import csv
import os
import sys
import time
from collections import deque

import torch
import torch.optim as optim

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_LOGDIR = os.path.join(_HERE, "logs")

from learned_scorer import ScoreNetwork
from strategic_reward import compute_strategic_score, dense_reward, average_strategic_metrics
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
    player_id: int = 0,
    strategic_scale: float = 0.01,
) -> dict:
    """Play one game, return dict with combined reward and strategic scores."""
    env.run([training_agent, baseline_agent])
    steps = env.steps
    final_reward = float(steps[-1][0]["reward"])

    strategic_scores = []
    for step_data in steps:
        obs = step_data[0].get("observation")
        if obs is None:
            break
        obs_tensors = single_obs_to_tensor(obs, player_id=player_id)
        planets = obs_tensors["planets"]
        score = compute_strategic_score(planets, player_id=player_id)
        strategic_scores.append(score)

    if len(strategic_scores) == len(steps):
        total_dense = torch.tensor(0.0)
        prev = None
        for s in strategic_scores:
            total_dense = total_dense + dense_reward(s, prev)
            prev = s
        combined = float(final_reward) + strategic_scale * float(total_dense)
        return {
            "reward": combined,
            "final_reward": final_reward,
            "strategic_reward": float(total_dense),
            "strategic_scores": strategic_scores,
        }

    return {
        "reward": final_reward,
        "final_reward": final_reward,
        "strategic_reward": 0.0,
        "strategic_scores": [],
    }


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


def _open_csv(path: str, fieldnames: list[str]):
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    fh = open(path, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=fieldnames)
    if not exists:
        w.writeheader()
        fh.flush()
    return w, fh


def train(
    num_episodes: int = 1000,
    hidden: int = 32,
    lr: float = 1e-3,
    temperature: float = 1.0,
    save_every: int = 100,
    eval_every: int = 50,
    baseline_ema: float = 0.95,
    seed: int = 42,
    strategic_scale: float = 0.01,
) -> ScoreNetwork:
    """Run REINFORCE with a running-return baseline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ScoreNetwork(hidden=hidden).to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    from main import agent as baseline_agent_fn
    from kaggle_environments import make as kgl_make
    env = kgl_make("orbit_wars", configuration={"seed": seed}, debug=False)

    # --- CSV logging setup --------------------------------------------------
    os.makedirs(_LOGDIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    train_csv, train_fh = _open_csv(
        os.path.join(_LOGDIR, f"reinforce_train_{ts}.csv"),
        [
            "episode", "reward", "winloss", "strategic_reward",
            "avg_frontline_strength", "avg_backline_productivity",
            "avg_ship_ratio", "avg_planet_ratio",
            "loss", "advantage", "log_prob_sum", "actions",
            "running_baseline", "winrate", "mean_return",
        ],
    )
    ep_csv, ep_fh = _open_csv(
        os.path.join(_LOGDIR, f"reinforce_episodes_{ts}.csv"),
        ["episode", "final_reward", "strategic_reward", "combined_reward",
         "waves", "timestamp"],
    )
    eval_csv, eval_fh = _open_csv(
        os.path.join(_LOGDIR, f"reinforce_eval_{ts}.csv"),
        ["episode", "winrate", "n_games"],
    )
    # ------------------------------------------------------------------------

    running_baseline = 0.0
    recent_returns: deque[float] = deque(maxlen=100)
    t0 = time.time()

    for ep in range(num_episodes):
        runtime = ProducerLiteRuntime(
            scorer=model,
            stochastic=True,
            temperature=temperature,
        )
        training_agent = _make_agent(runtime, no_grad=False)
        baseline_agent = baseline_agent_fn

        env.reset()
        try:
            result = run_episode(
                env, training_agent, baseline_agent,
                seed=seed + ep if seed else None,
                strategic_scale=strategic_scale,
            )
        except Exception as exc:
            print(f"  \u26a0  episode {ep} failed: {exc}")
            continue

        combined_reward = result["reward"]
        final_reward = result.get("final_reward", combined_reward)
        strategic_reward = result.get("strategic_reward", 0.0)
        strategic_scores = result.get("strategic_scores", [])

        recent_returns.append(combined_reward)
        running_baseline = baseline_ema * running_baseline + (1 - baseline_ema) * combined_reward

        log_probs = runtime.memory.episode_log_probs
        if not log_probs:
            ep_csv.writerow({
                "episode": ep, "final_reward": final_reward,
                "strategic_reward": strategic_reward,
                "combined_reward": combined_reward,
                "waves": 0, "timestamp": time.time(),
            })
            ep_fh.flush()
            continue

        lp_tensor = torch.stack([lp.to(device) for lp in log_probs])
        advantage = combined_reward - running_baseline
        loss = -lp_tensor.sum() * advantage

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # --- strategic averages -------------------------------------------
        str_avg = average_strategic_metrics(strategic_scores) if strategic_scores else {}

        # --- per-episode CSV ----------------------------------------------
        ep_csv.writerow({
            "episode": ep, "final_reward": final_reward,
            "strategic_reward": strategic_reward,
            "combined_reward": combined_reward,
            "waves": len(log_probs), "timestamp": time.time(),
        })
        ep_fh.flush()

        # --- logging -------------------------------------------------------
        if ep % max(save_every, 1) == 0 or ep == num_episodes - 1:
            mean_ret = float(sum(recent_returns)) / max(len(recent_returns), 1)
            win_rate = sum(1 for r in recent_returns if r > 0e0) / max(len(recent_returns), 1)

            # Train CSV row
            train_csv.writerow({
                "episode": ep,
                "reward": f"{combined_reward:+.4f}",
                "winloss": f"{final_reward:+.1f}",
                "strategic_reward": f"{strategic_reward:+.4f}",
                "avg_frontline_strength": str_avg.get("avg_frontline_strength", ""),
                "avg_backline_productivity": str_avg.get("avg_backline_productivity", ""),
                "avg_ship_ratio": str_avg.get("avg_ship_ratio", ""),
                "avg_planet_ratio": str_avg.get("avg_planet_ratio", ""),
                "loss": f"{loss.item():.4f}",
                "advantage": f"{advantage:.4f}",
                "log_prob_sum": f"{float(lp_tensor.sum()):.4f}",
                "actions": len(log_probs),
                "running_baseline": f"{running_baseline:.4f}",
                "winrate": f"{win_rate:.4f}",
                "mean_return": f"{mean_ret:.4f}",
            })
            train_fh.flush()

            print(
                f"ep={ep:5d}  reward={combined_reward:+.1f}  "
                f"wl={final_reward:+.1f}  "
                f"baseline={running_baseline:+.3f}  "
                f"loss={loss.item():+.4f}  "
                f"win100={win_rate:.3f}  "
                f"meanR={mean_ret:+.3f}  "
                f"actions={len(log_probs)}  "
                f"strat={strategic_reward:+.2f}"
            )
            ckpt = os.path.join(_HERE, f"scorer_ep{ep}.pt")
            torch.save(model.state_dict(), ckpt)

        if ep % max(eval_every, 1) == 0:
            model.eval()
            eval_env = kgl_make(
                "orbit_wars",
                configuration={"seed": seed + 9999},
                debug=False,
            )
            wr = evaluate(model, eval_env, baseline_agent_fn, n_games=10)
            model.train()
            print(f"         eval_win_rate={wr:.3f}")

            eval_csv.writerow({"episode": ep, "winrate": f"{wr:.4f}", "n_games": 10})
            eval_fh.flush()

    train_fh.close()
    ep_fh.close()
    eval_fh.close()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="REINFORCE training for ScoreNetwork")
    parser.add_argument("--episodes", type=int, default=1000, help="number of training episodes")
    parser.add_argument("--hidden", type=int, default=32, help="MLP hidden size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--temperature", type=float, default=1.0, help="softmax temperature for exploration")
    parser.add_argument("--save-every", type=int, default=100, help="checkpoint interval")
    parser.add_argument("--eval-every", type=int, default=50, help="evaluation interval")
    parser.add_argument("--strategic-scale", type=float, default=0.01,
                        help="weight of dense strategic reward (default 0.01)")
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
        strategic_scale=args.strategic_scale,
    )
