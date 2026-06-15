"""
sniper.py — Nearest-planet sniper opponent agent (heuristic).

Uses compute_attack_angle from utils via ow_features.heuristic,
which also handles sun/obstacle checks, moving-target leading, and comets.
"""
import sys, os
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

import ow_features as F


def agent(obs):
    return F.heuristic(obs)
