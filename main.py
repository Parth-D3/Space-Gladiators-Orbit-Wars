"""
Orbit Wars - Nearest Planet Sniper Agent

A simple agent that captures the nearest unowned planet when it has
enough ships to guarantee the takeover.

Strategy:
  For each planet we own, find the closest planet we don't own.
  If we have more ships than the target's garrison, send exactly
  enough to capture it (garrison + 1). Otherwise, wait and accumulate.

Key concepts demonstrated:
  - Parsing the observation (planets, player ID)
  - Computing angles with atan2 for fleet direction
  - Sending moves as [from_planet_id, angle, num_ships]
"""

import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from utils import compute_attack_angle


def agent(obs):
    moves = []
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets

    # Parse into named tuples for readable field access:
    #   Planet(id, owner, x, y, radius, ships, production)
    #   owner == -1 means neutral, 0-3 are player IDs
    planets = [Planet(*p) for p in raw_planets]
    my_planets = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player]

    if not targets:
        return moves

    for mine in my_planets:
        # Find the nearest planet we don't own
        nearest = None
        min_dist = float("inf")
        for t in targets:
            dist = math.sqrt((mine.x - t.x) ** 2 + (mine.y - t.y) ** 2)
            if dist < min_dist:
                min_dist = dist
                nearest = t

        if nearest is None:
            continue

        # We need to send more ships than the target has to capture it.
        # Exactly target_ships + 1 guarantees the takeover.
        ships_needed = nearest.ships + 1

        # Only launch if we can afford it — otherwise keep accumulating
        if mine.ships >= ships_needed:
            angle = compute_attack_angle(
                mine, nearest, ships_needed, planets,
                obs.get("initial_planets", []),
                obs.get("angular_velocity", 0),
                obs.get("step", 0),
            )
            if angle != -1.0:
                moves.append([mine.id, angle, ships_needed])

    return moves
