#!/usr/bin/env python3
"""Hyperparameter sweep for ProducerLiteConfig.

Plays each config variant vs the fixed baseline agent and reports
sorted win rates.  Run with::

    python hyper_sweep.py [--games 30] [--minimal]
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time
from collections.abc import Callable

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kaggle_environments import make

from main import (
    ProducerLiteConfig,
    ProducerLiteMemory,
    _apply_phase_config,
    _config_for,
    agent as baseline_agent_fn,
    largest_initial_player_count,
    run_turn,
    single_obs_to_tensor,
    sparse_action_row_to_moves,
)


# ---------------------------------------------------------------------------
# Sweep definitions
# ---------------------------------------------------------------------------

NAMED_SWEEPS: list[tuple[str, dict]] = [
    # --- reference ---
    ("baseline", {}),
    # --- roi_threshold (default 1.5) ---
    ("roi_0.8",    {"roi_threshold": 0.8}),
    ("roi_1.0",    {"roi_threshold": 1.0}),
    ("roi_2.0",    {"roi_threshold": 2.0}),
    ("roi_3.0",    {"roi_threshold": 3.0}),
    # --- horizon (default 18) ---
    ("hzn_10",     {"horizon": 10}),
    ("hzn_14",     {"horizon": 14}),
    ("hzn_24",     {"horizon": 24}),
    ("hzn_30",     {"horizon": 30}),
    # --- min_ships_to_launch (default 4) ---
    ("min_ships_2", {"min_ships_to_launch": 2.0}),
    ("min_ships_6", {"min_ships_to_launch": 6.0}),
    ("min_ships_8", {"min_ships_to_launch": 8.0}),
    # --- max_waves_per_turn (default 7) ---
    ("waves_4",    {"max_waves_per_turn": 4}),
    ("waves_10",   {"max_waves_per_turn": 10}),
    # --- regroup ---
    ("no_regroup", {"enable_regroup": False}),
    # --- size_multipliers (default (0.5, 1.0)) ---
    ("sizes_2tier", {"size_multipliers": (0.5, 1.0)}),
    ("sizes_4tier", {"size_multipliers": (0.25, 0.50, 0.75, 1.0)}),
]

GRID_SWEEPS = [
    # roi_threshold × horizon interactions
    ("roi0.8_hzn10",  {"roi_threshold": 0.8, "horizon": 10}),
    ("roi0.8_hzn24",  {"roi_threshold": 0.8, "horizon": 24}),
    ("roi1.0_hzn10",  {"roi_threshold": 1.0, "horizon": 10}),
    ("roi1.0_hzn24",  {"roi_threshold": 1.0, "horizon": 24}),
    ("roi2.0_hzn10",  {"roi_threshold": 2.0, "horizon": 10}),
    ("roi2.0_hzn24",  {"roi_threshold": 2.0, "horizon": 24}),
    ("roi3.0_hzn10",  {"roi_threshold": 3.0, "horizon": 10}),
    ("roi3.0_hzn24",  {"roi_threshold": 3.0, "horizon": 24}),
]

# Sweep of combinations of the best-performing individual configs.
# Singles + pairs + triples + quad combinations (roi_1.0 and roi_3.0 are
# never combined since they conflict).
_SINGLES: list[tuple[str, dict]] = [
    ("sizes_2tier",  {"size_multipliers": (0.5, 1.0)}),
    ("roi_1.0",      {"roi_threshold": 1.0}),
    ("roi_3.0",      {"roi_threshold": 3.0}),
    ("min_ships_2",  {"min_ships_to_launch": 2.0}),
    ("waves_4",      {"max_waves_per_turn": 4}),
]

_PAIRS: list[tuple[str, dict]] = [
    ("sz+roi1",     {"size_multipliers": (0.5, 1.0), "roi_threshold": 1.0}),
    ("sz+roi3",     {"size_multipliers": (0.5, 1.0), "roi_threshold": 3.0}),
    ("sz+ships2",   {"size_multipliers": (0.5, 1.0), "min_ships_to_launch": 2.0}),
    ("sz+waves4",   {"size_multipliers": (0.5, 1.0), "max_waves_per_turn": 4}),
    ("roi1+ships2", {"roi_threshold": 1.0, "min_ships_to_launch": 2.0}),
    ("roi3+ships2", {"roi_threshold": 3.0, "min_ships_to_launch": 2.0}),
    ("roi1+waves4", {"roi_threshold": 1.0, "max_waves_per_turn": 4}),
    ("roi3+waves4", {"roi_threshold": 3.0, "max_waves_per_turn": 4}),
    ("ships2+waves4", {"min_ships_to_launch": 2.0, "max_waves_per_turn": 4}),
]

_TRIPLES: list[tuple[str, dict]] = [
    ("sz+roi1+ships2",   {"size_multipliers": (0.5, 1.0), "roi_threshold": 1.0, "min_ships_to_launch": 2.0}),
    ("sz+roi3+ships2",   {"size_multipliers": (0.5, 1.0), "roi_threshold": 3.0, "min_ships_to_launch": 2.0}),
    ("sz+roi1+waves4",   {"size_multipliers": (0.5, 1.0), "roi_threshold": 1.0, "max_waves_per_turn": 4}),
    ("sz+roi3+waves4",   {"size_multipliers": (0.5, 1.0), "roi_threshold": 3.0, "max_waves_per_turn": 4}),
    ("sz+ships2+waves4", {"size_multipliers": (0.5, 1.0), "min_ships_to_launch": 2.0, "max_waves_per_turn": 4}),
    ("roi1+ships2+waves4", {"roi_threshold": 1.0, "min_ships_to_launch": 2.0, "max_waves_per_turn": 4}),
    ("roi3+ships2+waves4", {"roi_threshold": 3.0, "min_ships_to_launch": 2.0, "max_waves_per_turn": 4}),
]

_QUAD: list[tuple[str, dict]] = [
    ("sz+roi1+ships2+waves4", {"size_multipliers": (0.5, 1.0), "roi_threshold": 1.0, "min_ships_to_launch": 2.0, "max_waves_per_turn": 4}),
    ("sz+roi3+ships2+waves4", {"size_multipliers": (0.5, 1.0), "roi_threshold": 3.0, "min_ships_to_launch": 2.0, "max_waves_per_turn": 4}),
]

COMBINATION_SWEEPS: list[tuple[str, dict]] = _SINGLES + _PAIRS + _TRIPLES + _QUAD

ALL_SWEEPS = NAMED_SWEEPS + GRID_SWEEPS + COMBINATION_SWEEPS


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def make_sweep_agent(overrides: dict) -> Callable:
    """Return an agent that plays with *overrides* applied to the base
    ``ProducerLiteConfig`` each turn."""

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


# ---------------------------------------------------------------------------
# Single-config evaluation
# ---------------------------------------------------------------------------

def evaluate_config(
    overrides: dict,
    *,
    n_games: int,
    base_seed: int,
) -> tuple[int, int, int, float]:
    """Play ``n_games`` vs baseline.  Returns ``(wins, losses, draws, avg_game_len)``."""
    agent = make_sweep_agent(overrides)
    wins = losses = draws = 0
    total_steps = 0

    for game_idx in range(n_games):
        seed = base_seed + game_idx
        env = make(
            "orbit_wars",
            configuration={"seed": seed},
            debug=False,
        )
        env.run([agent, baseline_agent_fn])
        steps = env.steps
        total_steps += len(steps)
        reward = float(steps[-1][0]["reward"])
        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1

    avg_len = total_steps / max(n_games, 1)
    return wins, losses, draws, avg_len


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    n_games = max(1, int(args.games))
    base_seed = int(args.seed)
    target_sweeps = ALL_SWEEPS
    if args.minimal:
        target_sweeps = NAMED_SWEEPS
    if args.combinations:
        target_sweeps = [("baseline", {})] + COMBINATION_SWEEPS

    print(f"\n  Sweeping {len(target_sweeps)} configs × {n_games} games each\n")
    print(f"  {'Config':<25s}  {'Win Rate':>8s}  {'W/L/D':<12s}  {'Len':>5s}  {'Time':>5s}")
    print(f"  {'─' * 25}  {'─' * 8}  {'─' * 12}  {'─' * 5}  {'─' * 5}")

    rows: list[tuple[str, float, int, int, int, float]] = []

    for name, overrides in target_sweeps:
        t0 = time.time()
        wins, losses, draws, avg_len = evaluate_config(
            overrides, n_games=n_games, base_seed=base_seed,
        )
        elapsed = time.time() - t0
        win_rate = wins / max(n_games, 1)
        rows.append((name, win_rate, wins, losses, draws, avg_len))

        bar = "█" * max(1, int(20 * win_rate)) + "░" * max(0, 20 - max(1, int(20 * win_rate)))
        print(
            f"  {name:<25s}  {win_rate:>7.1%}  "
            f"{wins}/{losses}/{draws:<8}  {avg_len:>4.0f}  "
            f"{elapsed:>4.0f}s  {bar}",
            flush=True,
        )

    baseline_wr = next((r[1] for r in rows if r[0] == "baseline"), 0.5)
    rows.sort(key=lambda r: r[1], reverse=True)

    print(f"\n  {'=' * 75}")
    print(f"  RESULTS — {n_games} games per config, sorted by win rate")
    print(f"  {'=' * 75}")
    print(f"  {'Config':<25s}  {'Win Rate':>8s}  {'W/L/D':<12s}  {'Len':>5s}  {'Δ vs base':>9s}")
    print(f"  {'─' * 25}  {'─' * 8}  {'─' * 12}  {'─' * 5}  {'─' * 9}")
    for name, wr, w, l, d, al in rows:
        delta = wr - baseline_wr
        delta_s = f"+{delta:.1%}" if delta >= 0 else f"{delta:.1%}"
        print(f"  {name:<25s}  {wr:>7.1%}  {w}/{l}/{d:<8}  {al:>4.0f}  {delta_s:>9s}")

    top = [r for r in rows if r[0] != "baseline"][:5]
    print(f"\n  Top-5 performers:")
    for rank, (name, wr, w, l, d, al) in enumerate(top, 1):
        delta = wr - baseline_wr
        print(f"    {rank}. {name:<23s}  {wr:.1%}  ({w}/{l}/{d})  Δ={delta:+.1%}  len={al:.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep ProducerLiteConfig hyperparameters")
    parser.add_argument("--games", type=int, default=30, help="games per config")
    parser.add_argument("--seed", type=int, default=42, help="base RNG seed")
    parser.add_argument("--minimal", action="store_true", help="only named sweeps (skip grid)")
    parser.add_argument("--combinations", action="store_true", help="run combination sweep of best configs")
    args = parser.parse_args()

    main(args)
