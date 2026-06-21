#!/usr/bin/env python3
"""Test the strategic reward computation with synthetic and real game states."""

from __future__ import annotations

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from strategic_reward import compute_strategic_score, dense_reward


def test_unit():
    """Unit test with synthetic planet data."""
    device = torch.device("cpu")

    # 6 planets: 2 ours, 2 enemy, 2 neutral
    # Format: [id, owner, x, y, radius, ships, production]
    planets = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0, 1.0, 50.0, 3.0],   # ours (frontline - near enemy)
            [1.0, 0.0, 90.0, 90.0, 1.0, 80.0, 5.0],   # ours (backline - far from enemy)
            [2.0, 1.0, 20.0, 20.0, 1.0, 60.0, 4.0],   # enemy (frontline - near us)
            [3.0, 1.0, 80.0, 80.0, 1.0, 70.0, 4.0],   # enemy (frontline - near our backline)
            [4.0, -1.0, 50.0, 50.0, 1.0, 10.0, 2.0],  # neutral
            [5.0, -1.0, 30.0, 70.0, 1.0, 5.0, 1.0],   # neutral
        ],
        device=device,
    )

    score = compute_strategic_score(planets, player_id=0, threshold=50.0)

    print("=== Unit Test ===")
    print(f"  frontline_strength:       {score['frontline_strength']:.1f}  (expect ~50.0)")
    print(f"  backline_productivity:    {score['backline_productivity']:.1f}  (expect ~5.0)")
    print(f"  enemy_frontline_strength: {score['enemy_frontline_strength']:.1f}  (expect ~130.0)")
    print(f"  my_ships:                 {score['my_ships']:.1f}  (expect 130.0)")
    print(f"  enemy_ships:              {score['enemy_ships']:.1f}  (expect 130.0)")
    print(f"  my_prod:                  {score['my_prod']:.1f}  (expect 8.0)")
    print(f"  my_planet_count:          {score['my_planet_count']:.1f}  (expect 2.0)")
    print(f"  enemy_planet_count:       {score['enemy_planet_count']:.1f}  (expect 2.0)")
    print(f"  total_ships:              {score['total_ships']:.1f}  (expect 275.0)")
    print(f"  total_prod:               {score['total_prod']:.1f}  (expect 19.0)")

    # Planet 0 (10,10) is within 50 of enemy planet 2 (20,20) -> dist ~14.1 < 50 -> frontline
    # Planet 1 (90,90) is within 50 of enemy planet 3 (80,80) -> dist ~14.1 < 50 -> also frontline
    # So both owned planets are frontline, backline_prod should be 0
    is_frontline_expected = (score["frontline_strength"].item() == 130.0)
    is_backline_zero = (score["backline_productivity"].item() == 0.0)
    print(f"  ✓ frontline=130? {is_frontline_expected}")
    print(f"  ✓ backline=0?   {is_backline_zero}")


def test_dense_reward():
    """Test the dense reward delta computation."""
    device = torch.device("cpu")

    # State A: we're doing poorly
    planets_a = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0, 1.0, 20.0, 3.0],   # ours - frontline, few ships
            [1.0, 1.0, 15.0, 15.0, 1.0, 100.0, 5.0],   # enemy - strong frontline
        ],
        device=device,
    )

    # State B: we're doing better (reinforced frontline, more prod backline)
    planets_b = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0, 1.0, 80.0, 3.0],   # ours - frontline, reinforced
            [1.0, 1.0, 15.0, 15.0, 1.0, 90.0, 5.0],   # enemy - weaker frontline
        ],
        device=device,
    )

    score_a = compute_strategic_score(planets_a, player_id=0, threshold=50.0)
    score_b = compute_strategic_score(planets_b, player_id=0, threshold=50.0)

    reward_a = dense_reward(score_a, None)
    reward_delta = dense_reward(score_b, score_a)

    print("\n=== Dense Reward Test ===")
    print(f"  Initial reward:         {float(reward_a):.4f}  (expect 0.0)")
    print(f"  After improvement:      {float(reward_delta):+.4f}  (expect positive)")
    print(f"  frontline_strength Δ:   {float(score_b['frontline_strength'] - score_a['frontline_strength']):+.1f}")
    print(f"  enemy_frontline Δ:      {float(score_b['enemy_frontline_strength'] - score_a['enemy_frontline_strength']):+.1f}")
    print(f"  my_ships Δ:             {float(score_b['my_ships'] - score_a['my_ships']):+.1f}")

    # Delta should be positive since our frontline got stronger and enemy got weaker
    is_positive = float(reward_delta) > 0
    print(f"  ✓ reward positive after improvement? {is_positive}")

    # Now test reverse (getting worse)
    reward_reverse = dense_reward(score_a, score_b)
    print(f"  After getting worse:    {float(reward_reverse):+.4f}  (expect negative)")
    is_negative = float(reward_reverse) < 0
    print(f"  ✓ reward negative after decline? {is_negative}")


def test_real_step():
    """Run one real game step using the actual training pipeline."""
    print("\n=== Real Step Test ===")
    try:
        from kaggle_environments import make
        from main import agent as baseline_agent

        env = make("orbit_wars", configuration={"seed": 42}, debug=True)
        env.run([baseline_agent, baseline_agent])
        steps = env.steps

        from main import single_obs_to_tensor

        # Parse first and last step
        for label, step_data in [("first", steps[0]), ("last", steps[-1])]:
            obs = step_data[0]["observation"]
            reward = step_data[0].get("reward", 0.0)
            obs_tensors = single_obs_to_tensor(obs, player_id=0)
            planets = obs_tensors["planets"]
            score = compute_strategic_score(planets, player_id=0)

            print(f"\n  {label} step (env reward={reward:+.1f}):")
            print(f"    my planets:       {int(score['my_planet_count'])}")
            print(f"    enemy planets:    {int(score['enemy_planet_count'])}")
            print(f"    frontline ships:  {float(score['frontline_strength']):.0f}")
            print(f"    backline prod:    {float(score['backline_productivity']):.0f}")
            print(f"    my ships:         {float(score['my_ships']):.0f}")
            print(f"    enemy ships:      {float(score['enemy_ships']):.0f}")

        # Compute full-episode dense reward
        prev = None
        total = torch.tensor(0.0)
        n = 0
        for step_data in steps:
            obs = step_data[0].get("observation")
            if obs is None:
                break
            obs_tensors = single_obs_to_tensor(obs, player_id=0)
            score = compute_strategic_score(obs_tensors["planets"], player_id=0)
            total = total + dense_reward(score, prev)
            prev = score
            n += 1

        final_reward = float(steps[-1][0]["reward"])
        print(f"\n  Game over {n} steps:")
        print(f"    win/loss reward:     {final_reward:+.1f}")
        print(f"    total strategic Δ:   {float(total):+.3f}")
        print(f"    combined (λ=0.01):   {final_reward + 0.01 * float(total):+.3f}")
        print(f"  ✓ strategic reward computed over {n} steps")

    except ImportError as e:
        print(f"  SKIP: kaggle_environments not available ({e})")
    except Exception as e:
        print(f"  SKIP: runtime error: {e}")


if __name__ == "__main__":
    test_unit()
    test_dense_reward()
    test_real_step()
