#!/usr/bin/env python3
"""PPO self-play training for the learned scoring network.

Uses a clipped surrogate objective (PPO) with importance-sampling ratio,
a learned ValueNetwork for advantage estimation, self-play opponent pool
(past checkpoints), and entropy regularization.

Each episode stores per-selection (candidate_features, mask, selected_idx,
old_log_prob, state_features, old_value). During each PPO update we recompute
new log-probabilities from the stored candidate features via the current
ScoreNetwork to obtain the importance-sampling ratio π_new / π_old.

Usage
-----
    python train_ppo.py --episodes 2000 --lr 1e-3 --temperature 1.5
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_LOGDIR = os.path.join(_HERE, "logs")

from learned_scorer import ScoreNetwork, ValueNetwork
from strategic_reward import compute_strategic_score, dense_reward, average_strategic_metrics
from main import (
    ProducerLiteRuntime,
    single_obs_to_tensor,
    sparse_action_row_to_moves,
    agent as baseline_agent_fn,
)
from kaggle_environments import make


def _make_agent(runtime: ProducerLiteRuntime, *, no_grad: bool = True):
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
    env, training_agent, opponent_agent, *,
    seed: int | None = None, player_id: int = 0,
    strategic_scale: float = 0.01,
) -> dict:
    """Play one episode and return win/loss reward combined with strategic reward."""
    env.run([training_agent, opponent_agent])
    steps = env.steps
    final_reward = float(steps[-1][0]["reward"])

    # Compute dense strategic reward from per-step observations.
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


def evaluate(model: ScoreNetwork, value_net: ValueNetwork, env, *,
             n_games: int = 20) -> float:
    wins = 0
    for game in range(n_games):
        runtime = ProducerLiteRuntime(
            scorer=model, stochastic=False, value_network=value_net,
        )
        agent = _make_agent(runtime)
        env.run([agent, baseline_agent_fn])
        reward = float(env.steps[-1][0]["reward"])
        if reward > 0:
            wins += 1
    return wins / max(n_games, 1)


class OpponentPool:
    def __init__(self, max_size: int = 20, baseline: callable | None = None) -> None:
        self.max_size = max_size
        self.baseline = baseline
        self.checkpoints: list[str] = []

    def add(self, path: str) -> None:
        self.checkpoints.append(path)
        if len(self.checkpoints) > self.max_size:
            self.checkpoints.pop(0)

    def sample_opponent(self, model: ScoreNetwork, value_net: ValueNetwork,
                        device: torch.device) -> tuple[callable, str]:
        if not self.checkpoints:
            return self.baseline, "baseline"
        path = random.choice(self.checkpoints)
        ckpt = torch.load(path, map_location=device, weights_only=True)
        clone = ScoreNetwork(
            hidden=model.net[0].out_features,
        ).to(device)
        clone.load_state_dict(ckpt)
        clone.eval()
        runtime = ProducerLiteRuntime(scorer=clone, stochastic=False, value_network=value_net)
        return _make_agent(runtime), path


def _open_csv(path: str, fieldnames: list[str]):
    """Open a CSV file, write header if empty, return writer and file handle."""
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    fh = open(path, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=fieldnames)
    if not exists:
        w.writeheader()
        fh.flush()
    return w, fh


def _write_episode_row(
    episode_csv_writer, episode_file,
    ep: int, result: dict, opp_name: str, waves: int,
):
    """Write one row to the per-episode CSV."""
    row = {
        "episode": ep,
        "final_reward": result.get("final_reward", 0.0),
        "strategic_reward": result.get("strategic_reward", 0.0),
        "combined_reward": result.get("reward", 0.0),
        "opponent": opp_name,
        "waves": waves,
        "timestamp": time.time(),
    }
    episode_csv_writer.writerow(row)
    episode_file.flush()


def train(
    num_episodes: int = 2000,
    hidden: int = 32,
    lr: float = 1e-3,
    temperature: float = 1.5,
    batch_size: int = 8,
    k_epochs: int = 3,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    clip_epsilon: float = 0.2,
    gamma: float = 0.99,
    save_every: int = 50,
    eval_every: int = 50,
    seed: int = 42,
    strategic_scale: float = 0.01,
) -> tuple[ScoreNetwork, ValueNetwork]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ScoreNetwork(hidden=hidden).to(device)
    model.train()
    value_net = ValueNetwork(hidden=hidden).to(device)
    value_net.train()

    optimizer = optim.Adam(list(model.parameters()) + list(value_net.parameters()), lr=lr)

    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    opponent_pool = OpponentPool(max_size=20, baseline=baseline_agent_fn)

    # --- CSV logging setup --------------------------------------------------
    os.makedirs(_LOGDIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    train_csv, train_fh = _open_csv(
        os.path.join(_LOGDIR, f"ppo_train_{ts}.csv"),
        [
            "update", "episode", "winrate", "mean_return", "mean_winloss",
            "mean_strategic", "avg_frontline_strength", "avg_backline_productivity",
            "avg_ship_ratio", "avg_planet_ratio", "policy_loss", "value_loss",
            "entropy", "total_loss", "gradient_norm", "explained_variance",
            "waves_total", "wins", "losses", "draws", "n_episodes", "sps", "lr",
        ],
    )
    ep_csv, ep_fh = _open_csv(
        os.path.join(_LOGDIR, f"ppo_episodes_{ts}.csv"),
        ["episode", "final_reward", "strategic_reward", "combined_reward",
         "opponent", "waves", "timestamp"],
    )
    eval_csv, eval_fh = _open_csv(
        os.path.join(_LOGDIR, f"ppo_eval_{ts}.csv"),
        ["episode", "winrate", "n_games"],
    )
    # ------------------------------------------------------------------------

    recent_returns: deque[float] = deque(maxlen=100)
    recent_winloss: deque[float] = deque(maxlen=100)
    best_eval_wr: float = 0.0
    episode_counter = 0
    update_counter = 0
    t0 = time.time()

    while episode_counter < num_episodes:
        # --- collect batch (keep episode boundaries) -------------------------
        episodes_data: list[dict] = []
        batch_waves = 0
        batch_strategic_scores: list[dict] = []

        for _ in range(batch_size):
            if episode_counter >= num_episodes:
                break

            opp_agent, opp_name = opponent_pool.sample_opponent(model, value_net, device)
            runtime = ProducerLiteRuntime(
                scorer=model, stochastic=True, temperature=temperature,
                value_network=value_net,
            )
            training_agent = _make_agent(runtime, no_grad=False)

            env.reset()
            try:
                result = run_episode(
                    env, training_agent, opp_agent, seed=seed + episode_counter,
                    strategic_scale=strategic_scale,
                )
            except Exception as exc:
                print(f"  \u26a0  ep {episode_counter} failed: {exc}")
                episode_counter += 1
                continue

            reward = result["reward"]
            final_reward = result.get("final_reward", reward)
            recent_returns.append(reward)
            recent_winloss.append(final_reward)

            ppo_entries = runtime.memory.episode_ppo_entries
            state_feats = runtime.memory.episode_state_features
            values = runtime.memory.episode_values

            if not ppo_entries:
                episode_counter += 1
                continue

            n_waves = len(ppo_entries)
            batch_waves += n_waves
            strategic_scores = result.get("strategic_scores", [])
            if strategic_scores:
                batch_strategic_scores.extend(strategic_scores)

            episodes_data.append({
                "reward": reward,
                "final_reward": final_reward,
                "strategic_reward": result.get("strategic_reward", 0.0),
                "ppo_entries": ppo_entries,
                "state_feats": state_feats,
                "values": values,
                "opponent": opp_name,
            })

            # Per-episode CSV row
            _write_episode_row(ep_csv, ep_fh, episode_counter, result, opp_name, n_waves)

            episode_counter += 1

        if not episodes_data:
            continue

        update_counter += 1

        # --- PPO epochs (over all collected entries) -------------------------
        per_epoch_metrics: list[dict] = []
        for ei in range(k_epochs):
            policy_losses: list[Tensor] = []
            value_losses: list[Tensor] = []
            entropies: list[Tensor] = []
            value_preds: list[float] = []
            value_targets: list[float] = []

            for episode in episodes_data:
                reward = episode["reward"]
                ppo_entries = episode["ppo_entries"]
                state_feats = episode["state_feats"]
                values = episode["values"]

                for (cand_feats, mask, sel_idx, old_lp), sf, old_v in zip(
                    ppo_entries, state_feats, values
                ):
                    cf = cand_feats.to(device).clone().detach_()
                    m = mask.to(device).clone().detach_()
                    lp = old_lp.to(device).clone().detach_()
                    sf_c = sf.to(device).clone().detach_()

                    # --- policy: PPO clipped surrogate -------------------
                    new_scores = model(cf)                          # [C]
                    masked = torch.where(
                        m,
                        new_scores,
                        torch.tensor(float("-inf"), device=device),
                    )
                    new_log_probs = torch.log_softmax(masked / temperature, dim=-1)
                    new_lp = new_log_probs[sel_idx]                 # scalar

                    ratio = torch.exp(new_lp - lp)                  # π_new / π_old
                    advantage = reward - old_v.item()               # MC advantage
                    adv_t = torch.tensor(advantage, device=device)

                    surr1 = ratio * adv_t
                    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv_t
                    policy_losses.append(-torch.min(surr1, surr2))

                    # --- value: MSE against Monte-Carlo return -----------
                    v_pred = value_net(sf_c.unsqueeze(0)).squeeze(0)  # scalar
                    ve = (v_pred - torch.tensor(reward, device=device)) ** 2
                    value_losses.append(ve)
                    value_preds.append(float(v_pred.detach()))
                    value_targets.append(float(reward))

                    # --- entropy bonus ------------------------------------
                    entropies.append(new_lp)

            if not policy_losses:
                continue

            policy_loss = torch.stack(policy_losses).mean()
            value_loss = torch.stack(value_losses).mean()
            entropy = torch.stack(entropies).mean()

            total_loss = (policy_loss + vf_coef * value_loss + ent_coef * entropy)

            optimizer.zero_grad()
            total_loss.backward(retain_graph=False)
            grad_norm = nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(value_net.parameters()),
                max_norm=0.5,
            )
            optimizer.step()

            per_epoch_metrics.append({
                "policy_loss": float(policy_loss),
                "value_loss": float(value_loss),
                "entropy": float(entropy),
                "total_loss": float(total_loss),
                "gradient_norm": float(grad_norm),
            })

        # --- compute aggregate metrics ---------------------------------------
        if not per_epoch_metrics:
            continue

        last_epoch = per_epoch_metrics[-1]
        mean_ret = sum(recent_returns) / max(len(recent_returns), 1)
        mean_wl = sum(recent_winloss) / max(len(recent_winloss), 1)
        winrate_pct = sum(1 for r in recent_winloss if r > 0) / max(len(recent_winloss), 1)
        n_wins = sum(1 for r in recent_winloss if r > 0)
        n_losses = sum(1 for r in recent_winloss if r < 0)
        n_draws = sum(1 for r in recent_winloss if r == 0)

        # Strategic metrics from batch
        str_avg = average_strategic_metrics(batch_strategic_scores) if batch_strategic_scores else {}
        mean_strat = sum(e["strategic_reward"] for e in episodes_data) / len(episodes_data)

        # Explained variance
        vp = torch.tensor(value_preds)
        vt = torch.tensor(value_targets)
        var_res = ((vp - vt) ** 2).mean()
        var_ret = vt.var()
        ev = float(1.0 - var_res / var_ret.clamp(min=1e-8))

        sps = episode_counter / (time.time() - t0 + 1e-8)

        # --- log to CSV ------------------------------------------------------
        row = {
            "update": update_counter,
            "episode": episode_counter,
            "winrate": winrate_pct,
            "mean_return": f"{mean_ret:.4f}",
            "mean_winloss": f"{mean_wl:.4f}",
            "mean_strategic": f"{mean_strat:.4f}",
            "avg_frontline_strength": str_avg.get("avg_frontline_strength", ""),
            "avg_backline_productivity": str_avg.get("avg_backline_productivity", ""),
            "avg_ship_ratio": str_avg.get("avg_ship_ratio", ""),
            "avg_planet_ratio": str_avg.get("avg_planet_ratio", ""),
            "policy_loss": f"{last_epoch['policy_loss']:.6f}",
            "value_loss": f"{last_epoch['value_loss']:.6f}",
            "entropy": f"{last_epoch['entropy']:.6f}",
            "total_loss": f"{last_epoch['total_loss']:.6f}",
            "gradient_norm": f"{last_epoch['gradient_norm']:.4f}",
            "explained_variance": f"{ev:.4f}",
            "waves_total": batch_waves,
            "wins": n_wins,
            "losses": n_losses,
            "draws": n_draws,
            "n_episodes": len(episodes_data),
            "sps": f"{sps:.0f}",
            "lr": f"{lr:.6f}",
        }
        train_csv.writerow(row)
        train_fh.flush()

        # --- stdout logging --------------------------------------------------
        print(
            f"upd={update_counter:3d}  ep={episode_counter:5d}  "
            f"win={winrate_pct:.2f}  R={mean_ret:+.3f}  "
            f"strat={mean_strat:+.2f}  "
            f"pi={last_epoch['policy_loss']:.3f}  "
            f"v={last_epoch['value_loss']:.3f}  "
            f"ent={last_epoch['entropy']:.2f}  "
            f"gn={last_epoch['gradient_norm']:.2f}  "
            f"waves={batch_waves}  "
            f"ev={ev:.2f}  "
            f"opp={episodes_data[0]['opponent'][:16]}  "
            f"{sps:.0f}ep/s"
        )

        # --- save checkpoint -------------------------------------------------
        if episode_counter % max(save_every, 1) < batch_size or True:
            ckpt_path = os.path.join(_HERE, f"scorer_ppo_ep{episode_counter}.pt")
            torch.save(model.state_dict(), ckpt_path)
            v_path = os.path.join(_HERE, f"value_ppo_ep{episode_counter}.pt")
            torch.save(value_net.state_dict(), v_path)
            opponent_pool.add(ckpt_path)

        # --- evaluate ---------------------------------------------------------
        if episode_counter % max(eval_every, 1) < batch_size:
            model.eval()
            value_net.eval()
            eval_env = make(
                "orbit_wars", configuration={"seed": seed + 9999}, debug=False,
            )
            wr = evaluate(model, value_net, eval_env, n_games=10)
            model.train()
            value_net.train()
            print(f"         eval_win_rate={wr:.3f}")

            # Log eval to CSV
            eval_csv.writerow({"episode": episode_counter, "winrate": f"{wr:.4f}", "n_games": 10})
            eval_fh.flush()

            if wr > best_eval_wr:
                best_eval_wr = wr
                torch.save(model.state_dict(), os.path.join(_HERE, "scorer_ppo_best.pt"))
                torch.save(value_net.state_dict(), os.path.join(_HERE, "value_ppo_best.pt"))

    # --- close CSV files ---------------------------------------------------
    train_fh.close()
    ep_fh.close()
    eval_fh.close()

    return model, value_net


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO self-play training")
    parser.add_argument("--episodes", type=int, default=500, help="training episodes")
    parser.add_argument("--hidden", type=int, default=32, help="MLP hidden size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--temperature", type=float, default=1.5, help="exploration temperature")
    parser.add_argument("--batch-size", type=int, default=4, help="episodes per update")
    parser.add_argument("--k-epochs", type=int, default=3, help="PPO epochs per batch")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="value loss coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="entropy bonus coefficient")
    parser.add_argument("--save-every", type=int, default=50, help="checkpoint interval")
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
        batch_size=args.batch_size,
        k_epochs=args.k_epochs,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        save_every=args.save_every,
        eval_every=args.eval_every,
        seed=args.seed,
        strategic_scale=args.strategic_scale,
    )
