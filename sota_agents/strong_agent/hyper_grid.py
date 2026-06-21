#!/usr/bin/env python3
"""Round-robin tournament: every config plays every other config.

Usage
-----
    python hyper_grid.py --games 1       # 1 game per pair, both sides swapped = 2 games total
    python hyper_grid.py --games 3       # 6 games per pair
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time

import torch
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kaggle_environments import make

from main import (
    ProducerLiteConfig,
    ProducerLiteMemory,
    _apply_phase_config,
    _config_for,
    largest_initial_player_count,
    run_turn,
    single_obs_to_tensor,
    sparse_action_row_to_moves,
)


GRID_CONFIGS: list[tuple[str, dict]] = [
    ("baseline",   {}),
    ("roi_0.8",    {"roi_threshold": 0.8}),
    ("hzn_24",     {"horizon": 24}),
    ("roi0.8_24",  {"roi_threshold": 0.8, "horizon": 24}),
    ("waves_10",   {"max_waves_per_turn": 10}),
    ("no_regroup", {"enable_regroup": False}),
    ("roi_3.0",    {"roi_threshold": 3.0}),
    ("hzn_10",     {"horizon": 10}),
]


def make_agent(overrides: dict):
    """Build an agent callable that applies *overrides* to the base config."""
    allowed = {f.name for f in dataclasses.fields(ProducerLiteConfig)}
    cleaned = {}
    for k, v in overrides.items():
        if k not in allowed:
            continue
        if k == "size_multipliers":
            v = tuple(float(x) for x in v)
        cleaned[k] = v

    memory = ProducerLiteMemory()

    def agent(obs):
        player_id = int(obs["player"]) if isinstance(obs, dict) else int(obs.player)
        obs_tensors = single_obs_to_tensor(obs, player_id=player_id)
        if bool((obs_tensors["step"] == 0).all()):
            memory.cached_player_count = None
        if memory.cached_player_count is None:
            memory.cached_player_count = largest_initial_player_count(obs_tensors)
        current_player = int(obs_tensors["player"].reshape(-1)[0].item())
        min_count = current_player + 1
        memory.cached_player_count = (
            4 if max(int(memory.cached_player_count), min_count) > 2 else 2
        )
        base = _config_for(memory.cached_player_count)
        base = dataclasses.replace(base, **cleaned) if cleaned else base
        step = int(obs_tensors["step"].reshape(-1)[0].item())
        config = _apply_phase_config(base, step)

        with torch.no_grad():
            sparse_row = run_turn(
                obs_tensors,
                config=config,
                player_count=int(memory.cached_player_count),
                memory=memory,
            )
        return sparse_action_row_to_moves(sparse_row, obs, player_id=player_id)
    return agent


def run_matchup(agent_a, agent_b, *, n_games: int, base_seed: int):
    """Play ``n_games × 2`` matches (swap sides each game).  Returns ``(wins_a, wins_b, draws)``."""
    wins_a = wins_b = draws = 0
    for g in range(n_games):
        seed = base_seed + g * 2
        for swap in (False, True):
            agents = [agent_b, agent_a] if swap else [agent_a, agent_b]
            env = make("orbit_wars", configuration={"seed": seed + (1 if swap else 0)}, debug=False)
            env.run(agents)
            reward = float(env.steps[-1][0]["reward"])
            if swap:
                reward = -reward
            if reward > 0:
                wins_a += 1
            elif reward < 0:
                wins_b += 1
            else:
                draws += 1
    return wins_a, wins_b, draws


def main() -> None:
    parser = argparse.ArgumentParser(description="Round-robin grid sweep: every config vs every other")
    parser.add_argument("--games", type=int, default=1, help="games per pair (both sides swapped)")
    parser.add_argument("--seed", type=int, default=42, help="base RNG seed")
    args = parser.parse_args()

    N = len(GRID_CONFIGS)
    names = [n for n, _ in GRID_CONFIGS]
    agents = [make_agent(ov) for _, ov in GRID_CONFIGS]

    # Pair indexer, excluding diagonals.
    pairs: list[tuple[int, int]] = [(i, j) for i in range(N) for j in range(i + 1, N)]
    grid: list[list[tuple[int, int, int] | None]] = [[None] * N for _ in range(N)]

    t0 = time.time()
    pbar = tqdm(pairs, desc="Matchups", unit="pair")
    for i, j in pbar:
        w_a, w_b, d = run_matchup(
            agents[i], agents[j],
            n_games=args.games,
            base_seed=args.seed + (i * N + j) * args.games * 2,
        )
        grid[i][j] = (w_a, w_b, d)
        grid[j][i] = (w_b, w_a, d)

        # Runtime estimate in description.
        elapsed = time.time() - t0
        done = pbar.n + 1
        remaining = pairs[pbar.n:]

    elapsed_total = time.time() - t0

    # Compute total win rates.
    totals = [0.0 for _ in range(N)]
    for i in range(N):
        total_wins = sum(grid[i][j][0] for j in range(N) if j != i)
        total_games = sum(
            grid[i][j][0] + grid[i][j][1] + grid[i][j][2]
            for j in range(N) if j != i
        )
        totals[i] = total_wins / max(total_games, 1)

    # Sort by win rate descending.
    order = sorted(range(N), key=lambda i: -totals[i])

    # ── Print matrix ─────────────────────────────────────────────────
    print(f"\n  Grid: {N} configs, {args.games} game(s) per pair × 2 sides = "
          f"{2 * args.games} games per pair, {len(pairs)} pairs")
    print(f"  Total games played: {len(pairs) * 2 * max(args.games, 1)}")
    print(f"  Elapsed: {elapsed_total:.0f}s\n")

    # Header
    header = f"{'':>12s}"
    for idx in order:
        header += f"{names[idx]:>11s}"
    header += f"  {'Win%':>6s}"
    print(header)

    # Rows
    for i in order:
        row = f"{names[i]:<12s}"
        for j in order:
            if i == j:
                row += f"{'  -   ':>11s}"
            else:
                w, l, d = grid[i][j]
                tot = w + l + d
                if tot > 0:
                    pct = w / tot
                    row += f"{w:>2d}/{l:<2d} {pct:.2f}".ljust(11)
                else:
                    row += f"{'':>11s}"
        row += f"  {totals[i]:>6.3f}"
        print(row)

    print(f"\n  {'=' * (12 + 12 * N + 4)}")
    print(f"  {'Final ranking by total win rate':^36s}")
    total_wins_arr = [0] * N
    for i in range(N):
        tw = sum(grid[i][j][0] for j in range(N) if j != i)
        total_wins_arr[i] = tw
    for rank, idx in enumerate(order, 1):
        print(f"    {rank}. {names[idx]:<12s}  {totals[idx]:.3f}  ({total_wins_arr[idx]} wins)")


if __name__ == "__main__":
    main()
