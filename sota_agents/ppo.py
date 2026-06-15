"""
ppo.py — Orbit Wars submission entry point (PPO policy, greedy inference).

Runs the policy trained by train_ppo.py using pure numpy (no torch needed at
inference). If ppo_weights.npz is missing or anything fails, it falls back to
the repo's nearest-planet-sniper heuristic so the agent never errors out.

Local test (per agents.md):
    from kaggle_environments import make
    env = make("orbit_wars", debug=True)
    env.run(["sota_agents/ppo.py", "sniper.py"])

Submit (multi-file bundle, per agents.md):
    tar -czf submission.tar.gz sota_agents/ ppo_weights.npz ow_features.py utils.py
    kaggle competitions submit orbit-wars -f submission.tar.gz -m "PPO v1"
"""

import sys, os
try:
    _DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _DIR = os.getcwd()
_PARENT = os.path.dirname(_DIR)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import numpy as np
import ow_features as F


def _find_weights():
    for d in (_PARENT, os.getcwd(), os.path.dirname(_PARENT)):
        p = os.path.join(d, "ppo_weights.npz")
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
        return F.greedy_moves(_policy, obs, repair=True, defend=True, overflow_cap=300)
    except Exception:
        return F.heuristic(obs)
