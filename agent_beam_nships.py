"""
agent_beam_nships.py — beams a random number of ships (1..20 or avail-1)
at the same orbiting planet every 5 turns.
"""
from agent_shoot_all import parse
from utils import compute_attack_angle, is_orbiting
import random


def agent(obs):
    player, step, planets, w = parse(obs)

    if step % 5 != 0:
        return []

    orbiting = [p for p in planets if is_orbiting(p) and p["owner"] != player]
    if not orbiting:
        return []
    target = orbiting[0]

    moves = []
    for src in [p for p in planets if p["owner"] == player]:
        avail = int(src["ships"])
        if avail < 2:
            continue
        max_send = min(20, avail - 1)
        n = random.randint(1, max_send)
        angle = compute_attack_angle(src, target, n, planets, w)
        if angle >= 0:
            moves.append([src["id"], float(angle), n])
    return moves
