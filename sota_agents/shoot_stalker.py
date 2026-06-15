"""
agent_shoot_stalker.py — keeps shooting one orbiting planet every 5 turns.

Picks the first non-owned orbiting planet and fires 15-ship fleets
at it from every owned planet that can reach it.
"""
import sys, os
try:
    _DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _DIR = os.getcwd()
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)
_PARENT = os.path.dirname(_DIR)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from shoot_all import parse
from utils import compute_attack_angle, is_orbiting


def agent(obs):
    player, step, planets, w, comet_paths = parse(obs)

    orbiting = [p for p in planets if is_orbiting(p) and p["owner"] != player]
    if not orbiting:
        return []
    target = orbiting[0]

    moves = []
    for src in [p for p in planets if p["owner"] == player and p["ships"] >= 15]:
        angle = compute_attack_angle(src, target, 15, planets, w, comet_paths=comet_paths)
        if angle is not None:
            moves.append([src["id"], float(angle), 15])
    return moves
