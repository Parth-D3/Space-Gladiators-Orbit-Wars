"""
agent_shoot_stalker.py — keeps shooting one orbiting planet every 5 turns.

Picks the first non-owned orbiting planet and fires 1-ship fleets
at it from every owned planet that can reach it.
"""
from agent_shoot_all import parse
from utils import compute_attack_angle, is_orbiting


def agent(obs):
    player, step, planets, w = parse(obs)

    if step % 5 != 0:
        return []

    orbiting = [p for p in planets if is_orbiting(p) and p["owner"] != player]
    if not orbiting:
        return []
    target = orbiting[0]

    moves = []
    for src in [p for p in planets if p["owner"] == player and p["ships"] >= 1]:
        angle = compute_attack_angle(src, target, 1, planets, w)
        if angle >= 0:
            moves.append([src["id"], float(angle), 1])
    return moves
