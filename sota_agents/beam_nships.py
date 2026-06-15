"""
agent_beam_nships.py — sends fleets of increasing size (50, 80, 110, ...)
to the first non-owned orbiting planet. Waits for enough ships to accumulate
on each source planet before sending.
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

_next_fleet = {}


def agent(obs):
    global _next_fleet
    player, step, planets, w, comet_paths = parse(obs)

    orbiting = [p for p in planets if is_orbiting(p) and p["owner"] != player]
    if not orbiting:
        _next_fleet.clear()
        return []
    target = orbiting[0]

    moves = []
    for src in [p for p in planets if p["owner"] == player]:
        sid = src["id"]
        required = _next_fleet.get(sid, 50)
        avail = int(src["ships"])
        if avail < required:
            continue
        angle = compute_attack_angle(src, target, required, planets, w, comet_paths=comet_paths)
        if angle is not None:
            moves.append([sid, float(angle), required])
            _next_fleet[sid] = required + 30

    owned_ids = {p["id"] for p in planets if p["owner"] == player}
    for sid in list(_next_fleet.keys()):
        if sid not in owned_ids:
            del _next_fleet[sid]

    return moves
