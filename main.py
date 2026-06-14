"""
main.py — Orbit Wars submission entry point (PPO policy, greedy inference).

Runs the policy trained by train_ppo.py using pure numpy (no torch needed at
inference). If ppo_mlp_weights.npz is missing or anything fails, it falls back to
the repo's nearest-planet-sniper heuristic so the agent never errors out.

Local test (per agents.md):
    from kaggle_environments import make
    env = make("orbit_wars", debug=True)
    env.run(["main.py", "sniper.py"])

Submit (multi-file bundle, per agents.md):
    tar -czf submission.tar.gz main.py ow_features.py ppo_mlp_weights.npz
    kaggle competitions submit orbit-wars -f submission.tar.gz -m "PPO v1"
"""

import os
import sys

# Kaggle loads agents by exec'ing the file, where __file__ may be undefined.
try:
    _DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _DIR = os.getcwd()
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

import numpy as np
import ow_features as F

def _find_weights():
    for d in (_DIR, os.getcwd(), os.path.dirname(_DIR)):
        p = os.path.join(d, "ppo_mlp_weights.npz")
        if os.path.exists(p):
            return p
    return None


_policy = None
_load_attempted = False


def _load_policy():
    global _policy, _load_attempted
    if _load_attempted:
        return
    _load_attempted = True
    try:
        p = _find_weights()
        if p is not None:
            _policy = F.NumpyPolicy(np.load(p))
    except Exception:
        _policy = None


def agent(obs):
    _load_policy()
    if _policy is None:
        return F.heuristic(obs)
    try:
        # greedy_moves = policy + action repair + auto-defend + overflow valve
        return F.greedy_moves(_policy, obs, repair=True, defend=True, overflow_cap=300)
    except Exception:
        return F.heuristic(obs)
