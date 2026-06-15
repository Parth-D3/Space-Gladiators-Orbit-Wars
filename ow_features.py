"""
ow_features.py — shared observation encoding, action decoding and pure-numpy
policy inference for an Orbit Wars PPO agent.

Everything here is derived from the rules documented in the repo's README.md
and agents.md:
  * observation: planets [id, owner, x, y, radius, ships, production],
    fleets [id, owner, x, y, angle, from_planet_id, ships], player,
    angular_velocity, comet_planet_ids, step, ...        (README: Observation Reference)
  * action: list of [from_planet_id, angle_radians, num_ships]  (README: Action Format)
  * board 100x100, sun at (50, 50) with radius 10; fleets crossing the sun
    are destroyed                                         (README: Board Layout)
  * planets rotate iff orbital_radius + planet_radius < 50, at a constant
    angular velocity given in the observation             (README: Planet Types)
  * fleet speed = 1 + (maxSpeed - 1) * (log(ships)/log(1000))^1.5,
    maxSpeed default 6.0                                  (README: Fleet Speed)

This module uses only numpy + the standard library, so the Kaggle submission
does NOT need torch at inference time.
"""

import math
import numpy as np

from utils import compute_attack_angle

# ----- constants from README.md (Board Layout / Configuration defaults) -----
BOARD = 100.0
CX, CY = 50.0, 50.0        # sun center
ROT_LIMIT = 50.0           # planets with orbital_radius + radius < 50 rotate

# ----- encoding sizes (fixed-size tensors for the neural net) -----
MAX_PLANETS = 64           # README: 20-40 planets + comets (groups of 4)
PLANET_F = 17              # per-planet features (incl. velocity vx, vy)
GLOBAL_F = 13              # global features
FRACS = (0.0, 0.25, 0.5, 0.75, 1.0)   # garrison fraction to send (0 = no-op)
NFRAC = len(FRACS)
NEAR_R = 15.0              # radius for "fleet ships near this planet" features

_LOG1K = math.log(1000.0)
_LOG10K = math.log(10000.0)


def get(obs, key, default=None):
    """Read a field from a kaggle observation (dict-like Struct or plain dict)."""
    if isinstance(obs, dict):
        v = obs.get(key, default)
    else:
        v = getattr(obs, key, default)
    return default if v is None else v


def is_orbiting(x, y, radius, is_comet):
    """README 'Planet Types': orbital_radius + planet_radius < 50 => rotates."""
    if is_comet:
        return False
    return math.hypot(x - CX, y - CY) + radius < ROT_LIMIT


def parse(obs):
    """Parse the raw observation into python structures."""
    player = int(get(obs, "player", 0))
    comet_ids = set(get(obs, "comet_planet_ids", []) or [])
    rows = []
    for p in (get(obs, "planets", []) or []):
        rows.append(dict(
            id=int(p[0]), owner=int(p[1]), x=float(p[2]), y=float(p[3]),
            radius=float(p[4]), ships=float(p[5]), prod=float(p[6]),
            comet=int(p[0]) in comet_ids,
        ))
    fleets = []
    for f in (get(obs, "fleets", []) or []):
        fleets.append(dict(
            id=int(f[0]), owner=int(f[1]), x=float(f[2]), y=float(f[3]),
            angle=float(f[4]), src=int(f[5]), ships=float(f[6]),
        ))
    w = float(get(obs, "angular_velocity", 0.0) or 0.0)
    step = float(get(obs, "step", 0) or 0)
    # Comet trajectories: paths[i][path_index] is the comet's CURRENT position
    # (verified against the env), so position t turns ahead is path[idx + t].
    comet_paths = {}
    for grp in (get(obs, "comets", []) or []):
        ids = grp["planet_ids"] if isinstance(grp, dict) else grp.planet_ids
        paths = grp["paths"] if isinstance(grp, dict) else grp.paths
        idx = grp["path_index"] if isinstance(grp, dict) else grp.path_index
        for i, pid in enumerate(ids):
            comet_paths[int(pid)] = (paths[i], int(idx))
    return player, rows, fleets, w, step, comet_paths


def encode(obs):
    """Encode an observation into fixed-size arrays for the policy network.

    Returns:
      P        [MAX_PLANETS, PLANET_F] float32 per-planet features
      G        [GLOBAL_F]              float32 global features
      src_mask [MAX_PLANETS] bool      valid launch sources (mine, ships >= 1)
      tgt_mask [MAX_PLANETS] bool      valid targets (any existing planet)
      aux      dict used by decode_action()
    """
    player, rows, fleets, w, step, comet_paths = parse(obs)
    rows = rows[:MAX_PLANETS]

    P = np.zeros((MAX_PLANETS, PLANET_F), dtype=np.float32)
    for i, r in enumerate(rows):
        mine = 1.0 if r["owner"] == player else 0.0
        enemy = 1.0 if (r["owner"] != player and r["owner"] != -1) else 0.0
        neutral = 1.0 if r["owner"] == -1 else 0.0
        near_e = 0.0
        near_m = 0.0
        for f in fleets:
            if math.hypot(f["x"] - r["x"], f["y"] - r["y"]) < NEAR_R:
                if f["owner"] == player:
                    near_m += f["ships"]
                else:
                    near_e += f["ships"]
        # Per-planet velocity (units/turn) so the policy can SEE motion rather
        # than infer it from the orbiting flag + trig. Orbiting planets:
        # tangential velocity derived from d/dt of (cx + r cos(a+wt), cy + r sin(a+wt)).
        # Comets: next path point minus current. Static: zero.
        vx = vy = 0.0
        if r["comet"]:
            pp = comet_paths.get(r["id"])
            if pp is not None:
                path, idx = pp
                if idx + 1 < len(path):
                    vx = float(path[idx + 1][0]) - float(path[idx][0])
                    vy = float(path[idx + 1][1]) - float(path[idx][1])
        elif is_orbiting(r["x"], r["y"], r["radius"], False):
            a = math.atan2(r["y"] - CY, r["x"] - CX)
            r_orb = math.hypot(r["x"] - CX, r["y"] - CY)
            vx = -w * r_orb * math.sin(a)
            vy = w * r_orb * math.cos(a)
        P[i] = (
            1.0, mine, enemy, neutral,
            r["x"] / BOARD, r["y"] / BOARD, r["radius"] / 3.0,
            math.log1p(max(r["ships"], 0.0)) / _LOG1K,
            min(r["ships"], 500.0) / 500.0,   # de-saturated: big stacks stay distinguishable
            r["prod"] / 5.0,
            1.0 if r["comet"] else 0.0,
            1.0 if is_orbiting(r["x"], r["y"], r["radius"], r["comet"]) else 0.0,
            math.hypot(r["x"] - CX, r["y"] - CY) / 70.71,
            math.log1p(near_e) / _LOG1K,
            math.log1p(near_m) / _LOG1K,
            vx / 4.0,                     # cometSpeed default 4 = fastest object
            vy / 4.0,
        )

    myp = [r for r in rows if r["owner"] == player]
    opp = [r for r in rows if r["owner"] != player and r["owner"] != -1]
    neu = [r for r in rows if r["owner"] == -1]
    myf = [f for f in fleets if f["owner"] == player]
    opf = [f for f in fleets if f["owner"] != player]
    G = np.array([
        len(myp) / 40.0, len(opp) / 40.0, len(neu) / 40.0,
        math.log1p(sum(r["ships"] for r in myp)) / _LOG10K,
        math.log1p(sum(r["ships"] for r in opp)) / _LOG10K,
        math.log1p(sum(f["ships"] for f in myf)) / _LOG10K,
        math.log1p(sum(f["ships"] for f in opf)) / _LOG10K,
        min(len(myf), 50) / 50.0, min(len(opf), 50) / 50.0,
        w / 0.05,                       # README: angular velocity in 0.025-0.05
        min(step, 500.0) / 500.0,       # README: 500-turn games
        sum(r["prod"] for r in myp) / 100.0,
        sum(r["prod"] for r in opp) / 100.0,
    ], dtype=np.float32)

    src_mask = np.zeros(MAX_PLANETS, dtype=bool)
    tgt_mask = np.zeros(MAX_PLANETS, dtype=bool)
    for i, r in enumerate(rows):
        tgt_mask[i] = True
        if r["owner"] == player and r["ships"] >= 1.0:
            src_mask[i] = True
    has_src = bool(src_mask.any())
    if not has_src:
        src_mask[0] = True   # sentinel keeps the distribution valid; decode no-ops
    if not tgt_mask.any():
        tgt_mask[0] = True

    aux = dict(player=player, rows=rows, w=w, has_src=has_src, comet_paths=comet_paths)
    return P, G, src_mask, tgt_mask, aux


def totals(obs):
    """(my_total, best_opponent_total): ships on owned planets + in fleets.

    Matches the README 'Scoring and Termination' definition of the final score.
    """
    player, rows, fleets, _, _, _ = parse(obs)
    mine = sum(r["ships"] for r in rows if r["owner"] == player)
    mine += sum(f["ships"] for f in fleets if f["owner"] == player)
    others = {}
    for r in rows:
        if r["owner"] not in (-1, player):
            others[r["owner"]] = others.get(r["owner"], 0.0) + r["ships"]
    for f in fleets:
        if f["owner"] != player:
            others[f["owner"]] = others.get(f["owner"], 0.0) + f["ships"]
    return mine, (max(others.values()) if others else 0.0)


def _aimed_move(s, t, n, w, comet_paths, rows):
    angle = compute_attack_angle(s, t, n, rows, w, comet_paths=comet_paths)
    if angle is None:
        return []
    return [[int(s["id"]), float(angle), int(n)]]


def overflow_attack(aux, cap=300, send_frac=0.5):
    """Anti-hoarding valve: any owned planet whose garrison exceeds `cap`
    MUST attack — fixing the observed freeze where big stacks stop firing
    (rare in training data, so the greedy policy goes no-op on them).

    Target order per the requested priority: smallest garrison first, static
    before moving on ties, preferring targets the strike can actually flip.
    Tries candidates in order until one has a clear, sun-safe lane.
    Active in training too, so the policy learns around it.
    """
    rows, w, player = aux["rows"], aux["w"], aux["player"]
    comet_paths = aux.get("comet_paths", {})
    moves = []
    targets = [r for r in rows if r["owner"] != player]
    if not targets:
        return moves
    for p in rows:
        if p["owner"] != player or p["ships"] <= cap:
            continue
        n = int(send_frac * p["ships"])
        if n < 1:
            continue

        def order_key(t):
            movingness = 1 if (t["comet"] or is_orbiting(t["x"], t["y"], t["radius"], False)) else 0
            can_flip = 0 if n > t["ships"] else 1
            d = math.hypot(t["x"] - p["x"], t["y"] - p["y"])
            return (can_flip, t["ships"], movingness, d)

        for t in sorted(targets, key=order_key):
            mv = _aimed_move(p, t, n, w, comet_paths, rows)
            if mv:
                moves.extend(mv)
                break
    return moves


def auto_defend(aux, threshold=10):
    """Optional scripted reinforcement: if an owned planet's garrison is below
    `threshold`, top it up with `threshold` ships from the richest owned planet
    (only when the donor holds at least 4x the threshold, so defense never
    drains an attacker). At most one such move per turn. Uses the same
    intercept/sun logic as normal launches, so it can reinforce moving planets.
    """
    rows, w, player = aux["rows"], aux["w"], aux["player"]
    comet_paths = aux.get("comet_paths", {})
    owned = [r for r in rows if r["owner"] == player and not r["comet"]]
    needy = [r for r in owned if r["ships"] < threshold]
    if not needy:
        return []
    donors = [r for r in owned if r["ships"] >= 4 * threshold]
    if not donors:
        return []
    target = min(needy, key=lambda r: r["ships"])
    donor = max(donors, key=lambda r: r["ships"])
    if donor["id"] == target["id"]:
        return []
    return _aimed_move(donor, target, threshold, w, comet_paths, rows)


def decode_action(aux, src_i, tgt_i, frac_i):
    """Turn a sampled (source slot, target slot, fraction) into a kaggle move list.

    Returns [] (no-op) when the combination is invalid or suicidal:
      * fraction 0 selected, no owned source, src == tgt, slot out of range
      * no intercept exists (e.g. the comet leaves the board first)
      * the flight path would cross the sun — the README states such fleets
        are destroyed, so we skip the launch. For MOVING targets the checked
        segment extends 30 units past the aim point, because a near-miss
        keeps flying and can die in the sun behind the target.
    """
    rows, w, player = aux["rows"], aux["w"], aux["player"]
    comet_paths = aux.get("comet_paths", {})
    if not aux["has_src"] or frac_i <= 0:
        return []
    if src_i >= len(rows) or tgt_i >= len(rows) or src_i == tgt_i:
        return []
    s, t = rows[src_i], rows[tgt_i]
    if s["owner"] != player or s["ships"] < 1.0:
        return []

    frac = FRACS[frac_i]
    avail = int(s["ships"])
    if frac >= 0.999:
        # "Send all" leaves a minimal defensive garrison so a single greedy
        # launch can't strand a planet at 0 ships (a common elimination cause).
        n = max(1, avail - 1) if avail >= 3 else avail
    else:
        n = int(frac * avail)
    n = max(1, min(n, avail))
    if n < 1:
        return []

    return _aimed_move(s, t, n, w, comet_paths, rows)


def heuristic(obs):
    """Fallback policy when no trained weights are available.

    The repo's 'nearest planet sniper' (main.py) upgraded with the same
    intercept-aware aiming used by the RL agent, without kaggle imports.
    """
    player, rows, fleets, w, _, comet_paths = parse(obs)
    moves = []
    mine = [r for r in rows if r["owner"] == player]
    targets = [r for r in rows if r["owner"] != player]
    if not targets:
        return moves
    for m in mine:
        near = min(targets, key=lambda t: math.hypot(m["x"] - t["x"], m["y"] - t["y"]))
        need = int(near["ships"]) + 1
        if m["ships"] >= need:
            moves.extend(_aimed_move(m, near, need, w, comet_paths, rows))
    return moves


# --------------------------- numpy inference --------------------------------

def _relu(x):
    return np.maximum(x, 0.0)


class NumpyPolicy:
    """Greedy (argmax) inference mirroring train_ppo.TorchPolicy exactly.

    Loads weights exported by train_ppo.py with np.savez(model.state_dict()).
    Depth-agnostic: it discovers each block's layers ("pe.0", "pe.2", ...)
    from the weight file, so torch-side depth/width changes can't desync it.
    """

    def __init__(self, weights):
        keys = weights.files if hasattr(weights, "files") else weights.keys()
        self.w = {k: np.asarray(weights[k], dtype=np.float32) for k in keys}
        self.blocks = {}
        for k in self.w:
            parts = k.split(".")
            if len(parts) == 3 and parts[2] == "weight":
                self.blocks.setdefault(parts[0], []).append(int(parts[1]))
        for b in self.blocks:
            self.blocks[b].sort()

    def _mlp(self, x, block, final_relu):
        idxs = self.blocks[block]
        for j, i in enumerate(idxs):
            x = x @ self.w[f"{block}.{i}.weight"].T + self.w[f"{block}.{i}.bias"]
            if j < len(idxs) - 1 or final_relu:
                x = _relu(x)
        return x

    def logits(self, P, G):
        e = self._mlp(P, "pe", True)                                    # [P, H]
        pres = P[:, 0:1]                                                # [P, 1]
        denom = max(float(pres.sum()), 1.0)
        mean = (e * pres).sum(axis=0) / denom
        mx = np.where(pres > 0, e, -1e9).max(axis=0)
        mx = np.where(mx < -1e8, 0.0, mx)
        ctx = self._mlp(np.concatenate([mean, mx, G]), "ctx", True)
        joint = np.concatenate([e, np.tile(ctx, (e.shape[0], 1))], axis=1)
        sl = self._mlp(joint, "src", False)[:, 0]
        tl = self._mlp(joint, "tgt", False)[:, 0]
        fl = self._mlp(ctx, "frac", False)
        return sl, tl, fl

    def act(self, P, G, src_mask, tgt_mask):
        sl, tl, fl = self.logits(P, G)
        sl = np.where(src_mask, sl, -1e9)
        tl = np.where(tgt_mask, tl, -1e9)
        return int(sl.argmax()), int(tl.argmax()), int(fl.argmax())




def greedy_moves(policy, obs, repair=True, defend=True, overflow_cap=300):
    """Full greedy turn used by BOTH evaluation and the Kaggle submission.

    1. Encode and run the policy.
    2. If the chosen fraction is the deliberate no-op, respect it.
    3. Otherwise ACTION REPAIR: greedy argmax is deterministic, so a blocked
       launch (sun lane, occupied lane, expiring comet) would silently freeze
       the agent forever. Instead, walk the policy's own preference ranking
       (top sources x top targets x top fractions) until a move decodes.
    4. Append auto_defend and overflow_attack moves.
    """
    P, G, sm, tm, aux = encode(obs)
    moves = []
    if aux["has_src"]:
        sl, tl, fl = policy.logits(P, G)
        sl = np.where(sm, sl, -1e9)
        tl = np.where(tm, tl, -1e9)
        f_order = list(np.argsort(-fl))
        if f_order[0] != 0:
            s_order = list(np.argsort(-sl))[:3]
            t_order = list(np.argsort(-tl))[:6]
            fracs = [f for f in f_order if f != 0][:2] if repair else [f_order[0]]
            found = False
            for f_i in fracs:
                for s_i in s_order:
                    if sl[s_i] <= -1e8:
                        break
                    for t_i in t_order:
                        if tl[t_i] <= -1e8:
                            break
                        mv = decode_action(aux, int(s_i), int(t_i), int(f_i))
                        if mv:
                            moves.extend(mv)
                            found = True
                            break
                    if found:
                        break
                if found or not repair:
                    break
    if defend:
        moves = moves + auto_defend(aux)
    if overflow_cap and overflow_cap > 0:
        moves = moves + overflow_attack(aux, cap=overflow_cap)
    return moves
