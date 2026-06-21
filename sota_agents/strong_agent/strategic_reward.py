"""Strategic dense reward computation for the strong_agent.

Computes per-step strategic state evaluations (frontline strength, backline
productivity, ship/planet advantage) and produces dense intermediate rewards
via potential-based reward shaping.  This gives the training signal richer
information than a sparse terminal win/loss alone.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def compute_strategic_score(
    planets: Tensor,
    player_id: int,
    *,
    threshold: float = 50.0,
) -> dict[str, Tensor | float]:
    """Compute strategic state metrics from a ``[P, 7]`` planet observation.

    Each planet row: ``[id, owner, x, y, radius, ships, production]``.

    Returns a flat dict of scalar metrics:
        frontline_strength, backline_productivity,
        enemy_frontline_strength,
        my_ships, enemy_ships, my_prod,
        my_planet_count, enemy_planet_count,
        total_ships, total_prod, total_alive
    """
    alive = planets[:, 0] >= 0
    owner = planets[:, 1].long()
    ships = planets[:, 5].float()
    prod = planets[:, 6].float()
    px = planets[:, 2]
    py = planets[:, 3]

    my_mask = (owner == player_id) & alive
    enemy_mask = (owner >= 0) & (owner != player_id) & alive

    total_alive = alive.sum().float()
    total_ships = ships[alive].sum().clamp(min=1.0)
    total_prod = prod[alive].sum().clamp(min=1.0)

    my_ships = ships[my_mask].sum() if my_mask.any() else torch.tensor(0.0, device=planets.device)
    enemy_ships = (
        ships[enemy_mask].sum() if enemy_mask.any() else torch.tensor(0.0, device=planets.device)
    )
    my_prod = prod[my_mask].sum() if my_mask.any() else torch.tensor(0.0, device=planets.device)
    n_my = my_mask.sum().float()
    n_enemy = enemy_mask.sum().float()

    frontline_ships = torch.tensor(0.0, device=planets.device)
    backline_prod = torch.tensor(0.0, device=planets.device)
    enemy_frontline_ships = torch.tensor(0.0, device=planets.device)

    my_idx = torch.where(my_mask)[0]
    enemy_idx = torch.where(enemy_mask)[0]

    if my_idx.numel() > 0 and enemy_idx.numel() > 0:
        dx = px.unsqueeze(1) - px.unsqueeze(0)
        dy = py.unsqueeze(1) - py.unsqueeze(0)
        dist = torch.sqrt((dx * dx + dy * dy).clamp(min=0.0))

        my_to_enemy = dist[my_idx][:, enemy_idx]
        min_dist = my_to_enemy.min(dim=1).values

        is_frontline = min_dist <= threshold
        is_backline = ~is_frontline

        if is_frontline.any():
            frontline_ships = ships[my_idx][is_frontline].sum()
        if is_backline.any():
            backline_prod = prod[my_idx][is_backline].sum()

        enemy_to_my = dist[enemy_idx][:, my_idx]
        e_min_dist = enemy_to_my.min(dim=1).values
        e_front = e_min_dist <= threshold
        if e_front.any():
            enemy_frontline_ships = ships[enemy_idx][e_front].sum()

    return {
        "frontline_strength": frontline_ships,
        "backline_productivity": backline_prod,
        "enemy_frontline_strength": enemy_frontline_ships,
        "my_ships": my_ships,
        "enemy_ships": enemy_ships,
        "my_prod": my_prod,
        "my_planet_count": n_my,
        "enemy_planet_count": n_enemy,
        "total_ships": total_ships,
        "total_prod": total_prod,
        "total_alive": total_alive,
    }


def average_strategic_metrics(
    scores: list[dict[str, Tensor | float]],
) -> dict[str, float]:
    """Average per-step strategic scores over an episode.

    Returns flat dict of mean values for all tracked metrics.
    """
    if not scores:
        return {}
    keys = list(scores[0].keys())
    avg = {}
    for k in keys:
        vals = []
        for s in scores:
            v = s[k]
            vals.append(float(v) if torch.is_tensor(v) else float(v))
        avg[f"avg_{k}"] = sum(vals) / len(vals)
    return avg


def dense_reward(
    current: dict[str, Tensor | float],
    previous: dict[str, Tensor | float] | None,
) -> Tensor:
    """Potential-based dense reward from the change in strategic state.

    Returns a scalar reward.  Returns 0 when *previous* is ``None`` (first step).
    """
    if previous is None:
        dev = (
            current["frontline_strength"].device
            if torch.is_tensor(current["frontline_strength"])
            else "cpu"
        )
        return torch.tensor(0.0, device=dev)

    def _val(d: dict, k: str) -> Tensor:
        v = d[k]
        return v if torch.is_tensor(v) else torch.tensor(v)

    def _safe_div(num: Tensor, den: Tensor) -> Tensor:
        return num / den.clamp(min=1.0)

    # Frontline strength ratio — our front-line ships per enemy front-line ship
    cur_fl = _safe_div(_val(current, "frontline_strength"), _val(current, "enemy_frontline_strength"))
    prev_fl = _safe_div(_val(previous, "frontline_strength"), _val(previous, "enemy_frontline_strength"))

    # Backline production share
    cur_bp = _safe_div(_val(current, "backline_productivity"), _val(current, "total_prod"))
    prev_bp = _safe_div(_val(previous, "backline_productivity"), _val(previous, "total_prod"))

    # Ship advantage (normalised by total alive ships)
    cur_sa = _safe_div(
        _val(current, "my_ships") - _val(current, "enemy_ships"), _val(current, "total_ships")
    )
    prev_sa = _safe_div(
        _val(previous, "my_ships") - _val(previous, "enemy_ships"), _val(previous, "total_ships")
    )

    # Planet advantage (normalised by alive planets)
    cur_pa = _safe_div(
        _val(current, "my_planet_count") - _val(current, "enemy_planet_count"),
        _val(current, "total_alive"),
    )
    prev_pa = _safe_div(
        _val(previous, "my_planet_count") - _val(previous, "enemy_planet_count"),
        _val(previous, "total_alive"),
    )

    reward = (
        0.5 * (cur_fl - prev_fl)
        + 0.5 * (cur_bp - prev_bp)
        + 2.0 * (cur_sa - prev_sa)
        + 1.0 * (cur_pa - prev_pa)
    )
    return reward
