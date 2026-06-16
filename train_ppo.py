"""
train_ppo.py — Proximal Policy Optimization (Schulman et al. 2017,
arXiv:1707.06347) for Orbit Wars.

Default: trains against the repo's "nearest planet sniper" agent (sniper.py)
through kaggle_environments' trainer interface (env.train).

With --selfplay: trains against a greedy (argmax) copy of the current policy,
optionally snapshotting the model every N updates for evaluation.

Action model (one decision per turn):
    source planet slot  (masked to planets you own with >= 1 ship)
  x target planet slot  (masked to existing planets; own planets allowed
                         -> reinforcement)
  x garrison fraction   {no-op, 25%, 50%, 75%, 100%}

Reward (computed from observations, see ow_features.totals):
    base:   delta(my_total_ships - opp_total_ships) / 200     per step
    bonus:  +0.25 + 0.05*prod    per planet captured          one-shot
            -0.25 - 0.05*prod - garrison/100   per planet lost (non-comet)  one-shot
            -ships / 50   per comet garrison lost to expiry   one-shot
            -ships * max(0, 1 - turns/5) / 200  per launch at expiring comet  one-shot
            +0.001 * delta(prod_advantage)       per step
    terminal:  +3 win / -3 loss / 0 draw   (README scoring: highest total
               ships on planets + fleets wins; elimination also ends games)

Usage:
    python train_ppo.py --total-steps 300000                      # train vs sniper
    python train_ppo.py --total-steps 300000 --selfplay           # train via self-play
    python train_ppo.py --eval 20                                 # evaluate exported npz
    python train_ppo.py --eval 20 --selfplay                      # eval self-play snapshot
    python train_ppo.py --resume ppo_orbitwars.pt ...             # continue training
"""

import argparse
import collections
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from kaggle_environments import make

import ow_features as F

try:
    import wandb as _wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False


# ------------------------------ policy net ----------------------------------

def _mlp(in_dim, hidden, out_dim, n_hidden):
    """n_hidden hidden layers of uniform width `hidden`, then a linear output.
    Param names come out as block.0, block.2, ... which NumpyPolicy discovers."""
    layers = [nn.Linear(in_dim, hidden)]
    for _ in range(n_hidden - 1):
        layers += [nn.ReLU(), nn.Linear(hidden, hidden)]
    layers += [nn.ReLU(), nn.Linear(hidden, out_dim)]
    return nn.Sequential(*layers)


class TorchPolicy(nn.Module):
    """Per-planet encoder + pooled context; heads for source/target/fraction.

    v4 architecture: each block has 2 more layers than v3, and every hidden
    layer is the old width + 32 (uniform within a block):
      planet encoder & heads: 64 -> 96 wide, 2 -> 4 linears
      context block:         128 -> 160 wide, 2 -> 4 linears
    Layer names/shapes mirror ow_features.NumpyPolicy (which is depth-agnostic)
    so the exported .npz reproduces the trained policy exactly at inference.
    """

    H = 96    # per-planet embedding width (was 64)
    C = 160   # context width (was 128)

    def __init__(self):
        super().__init__()
        H, C = self.H, self.C
        self.pe = _mlp(F.PLANET_F, H, H, n_hidden=3)
        self.ctx = _mlp(2 * H + F.GLOBAL_F, C, C, n_hidden=3)
        self.src = _mlp(H + C, H, 1, n_hidden=3)
        self.tgt = _mlp(H + C, H, 1, n_hidden=3)
        self.frac = _mlp(C, H, F.NFRAC, n_hidden=3)
        self.val = _mlp(C, H, 1, n_hidden=3)

    def trunk(self, P, G):
        # P: [B, Np, PLANET_F]   G: [B, GLOBAL_F]
        e = torch.relu(self.pe(P))                                   # [B, Np, H]
        pres = P[..., 0:1]                                           # [B, Np, 1]
        denom = pres.sum(dim=1).clamp(min=1.0)                       # [B, 1]
        mean = (e * pres).sum(dim=1) / denom                         # [B, H]
        mx = e.masked_fill(pres <= 0, -1e9).max(dim=1).values
        mx = torch.where(mx < -1e8, torch.zeros_like(mx), mx)        # [B, H]
        ctx = torch.relu(self.ctx(torch.cat([mean, mx, G], dim=-1)))  # [B, C]
        joint = torch.cat([e, ctx.unsqueeze(1).expand(-1, e.size(1), -1)], dim=-1)
        src_l = self.src(joint).squeeze(-1)                          # [B, Np]
        tgt_l = self.tgt(joint).squeeze(-1)                          # [B, Np]
        frac_l = self.frac(ctx)                                      # [B, NFRAC]
        value = self.val(ctx).squeeze(-1)                            # [B]
        return src_l, tgt_l, frac_l, value

    def dists(self, P, G, src_mask, tgt_mask):
        src_l, tgt_l, frac_l, value = self.trunk(P, G)
        src_l = src_l.masked_fill(~src_mask, -1e9)
        tgt_l = tgt_l.masked_fill(~tgt_mask, -1e9)
        Cat = torch.distributions.Categorical
        return Cat(logits=src_l), Cat(logits=tgt_l), Cat(logits=frac_l), value


# ---------------------------- self-play opponent ----------------------------

def _greedy_opponent(model, device, auto_defend=True, overflow_cap=300):
    """Return an agent(obs) callable using TorchPolicy with greedy argmax
    + action repair (same logic as F.greedy_moves)."""
    @torch.no_grad()
    def agent(obs):
        P, G, sm, tm, aux = F.encode(obs)
        if not aux["has_src"]:
            return []

        tP = torch.as_tensor(P, device=device).unsqueeze(0)
        tG = torch.as_tensor(G, device=device).unsqueeze(0)
        src_l, tgt_l, frac_l, _ = model.trunk(tP, tG)

        sl = src_l.squeeze(0).cpu().numpy()
        tl = tgt_l.squeeze(0).cpu().numpy()
        fl = frac_l.squeeze(0).cpu().numpy()

        sl = np.where(sm, sl, -1e9)
        tl = np.where(tm, tl, -1e9)

        moves = []
        f_order = list(np.argsort(-fl))
        if f_order[0] != 0:
            s_order = list(np.argsort(-sl))[:3]
            t_order = list(np.argsort(-tl))[:6]
            fracs = [f for f in f_order if f != 0][:2]
            found = False
            for f_i in fracs:
                for s_i in s_order:
                    if sl[s_i] <= -1e8:
                        break
                    for t_i in t_order:
                        if tl[t_i] <= -1e8:
                            break
                        mv = F.decode_action(aux, int(s_i), int(t_i), int(f_i))
                        if mv:
                            moves.extend(mv)
                            found = True
                            break
                    if found:
                        break
                if found:
                    break

        if auto_defend:
            moves += F.auto_defend(aux)
        if overflow_cap:
            moves += F.overflow_attack(aux, cap=overflow_cap)
        return moves

    return agent


# --------------------------- strong-agent opponent ---------------------------

def _make_strong_opponent():
    """Return an agent(obs) callable using a fresh ProducerLiteRuntime.
    Each call creates a clean runtime so state does not leak between episodes."""
    from sota_agents.strong_agent.main import ProducerLiteRuntime
    from sota_agents.strong_agent.orbit_lite.adapter import (
        single_obs_to_tensor,
        sparse_action_row_to_moves,
    )
    runtime = ProducerLiteRuntime()
    def agent(obs):
        player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
        player_id = int(player)
        obs_tensors = single_obs_to_tensor(obs, player_id=player_id)
        with torch.no_grad():
            sparse_row = runtime.tensor_action(obs_tensors)
        return sparse_action_row_to_moves(sparse_row, obs, player_id=player_id)
    return agent


# ------------------------------ environment ---------------------------------

class Game:
    """One Orbit Wars game: us (RL) vs 1 or 3 copies of an opponent agent file.

    Uses kaggle_environments' trainer interface (agents.md: env.train).
    The env spec supports 2 or 4 players. `players` may be 2, 4, or "mix"
    (random per episode) so the policy learns both 1v1 and 1v3 dynamics.
    Seats are randomized per episode; the board is mirror-symmetric per the
    README, but seat randomization also varies which quadrant we start in.
    """

    def __init__(self, opponent, seat=None, players="mix"):
        self.opponent = opponent
        self.seat = seat   # None = random, else fixed seat index
        self.players = players

    def reset(self):
        if self.players in (2, 4):
            n = self.players
        else:
            n = random.choice([2, 4])
        self.num_players = n
        seat = self.seat if self.seat is not None else random.randrange(n)
        seat = seat % n
        agents = [self.opponent] * n
        agents[seat] = None
        self.env = make("orbit_wars", debug=False)
        self.trainer = self.env.train(agents)
        obs = self.trainer.reset()
        m, o = F.totals(obs)
        self.prev_diff = m - o
        self.last_obs = obs
        return obs

    def step(self, moves):
        obs, _env_reward, done, _info = self.trainer.step(moves)
        extra = 0.0

        # Detect events between steps for reward shaping
        if self.last_obs is not None:
            pv, pr, _, _, _, _ = F.parse(self.last_obs)
            cv, cr, _, _, _, _ = F.parse(obs)

            # 1. Comet garrison loss penalty (scaled by ships lost)
            prev_cids = {r["id"] for r in pr if r["comet"]}
            curr_cids = {r["id"] for r in cr if r["comet"]}
            ships_lost = sum(r["ships"] for r in pr
                             if r["id"] in (prev_cids - curr_cids)
                             and r["owner"] == pv)
            if ships_lost > 0:
                extra -= ships_lost / 50.0

            # 2. Planet capture bonus
            prev_own = {r["id"] for r in pr if r["owner"] == pv}
            curr_own = {r["id"] for r in cr if r["owner"] == cv}
            for r in cr:
                if r["id"] in (curr_own - prev_own):
                    extra += 0.25 + 0.05 * r["prod"]

            # 3. Planet loss penalty (non-comet — comets handled above)
            for r in pr:
                if r["id"] in (prev_own - curr_own) and not r["comet"]:
                    extra -= 0.25 + 0.05 * r["prod"] + r["ships"] / 100.0

            # 4. Production appreciation (leading indicator)
            pmp = sum(r["prod"] for r in pr if r["owner"] == pv)
            pop = sum(r["prod"] for r in pr if r["owner"] not in (-1, pv))
            cmp_ = sum(r["prod"] for r in cr if r["owner"] == cv)
            cop = sum(r["prod"] for r in cr if r["owner"] not in (-1, cv))
            extra += 0.001 * ((cmp_ - cop) - (pmp - pop))

        m, o = F.totals(obs)
        diff = m - o
        reward = (diff - self.prev_diff) / 200.0 + extra
        self.prev_diff = diff
        self.last_obs = obs
        outcome = 0
        if done:
            outcome = 1 if m > o else (-1 if m < o else 0)
            reward += 3.0 * outcome
        return obs, reward, done, outcome


class SelfPlayGame(Game):
    """Self-play: the current policy (eval mode, greedy argmax) is the opponent.
    The model reference is shared so both sides use the same weights within
    a rollout.  The learner samples (stochastic); the opponent takes argmax
    (deterministic, same as F.greedy_moves)."""

    def __init__(self, model, device, seat=None, players="mix",
                 auto_defend=True, overflow_cap=300):
        self.model = model
        self.device = device
        self.seat = seat
        self.players = players
        self.auto_defend = auto_defend
        self.overflow_cap = overflow_cap
        self._opp_fn = _greedy_opponent(model, device, auto_defend, overflow_cap)

    def reset(self):
        if self.players in (2, 4):
            n = self.players
        else:
            n = random.choice([2, 4])
        self.num_players = n
        seat = self.seat if self.seat is not None else random.randrange(n)
        seat = seat % n
        agents = [self._opp_fn] * n
        agents[seat] = None
        self.env = make("orbit_wars", debug=False)
        self.trainer = self.env.train(agents)
        obs = self.trainer.reset()
        m, o = F.totals(obs)
        self.prev_diff = m - o
        self.last_obs = obs
        return obs


class StrongAgentGame(Game):
    """Train against the torch-based strong agent (sota_agents/strong_agent).
    A fresh runtime is created per episode so that movement/launch memory
    does not leak between games."""

    def __init__(self, seat=None, players="mix"):
        self.seat = seat
        self.players = players

    def reset(self):
        if self.players in (2, 4):
            n = self.players
        else:
            n = random.choice([2, 4])
        self.num_players = n
        seat = self.seat if self.seat is not None else random.randrange(n)
        seat = seat % n
        opp_fn = _make_strong_opponent()
        agents = [opp_fn] * n
        agents[seat] = None
        self.env = make("orbit_wars", debug=False)
        self.trainer = self.env.train(agents)
        obs = self.trainer.reset()
        m, o = F.totals(obs)
        self.prev_diff = m - o
        self.last_obs = obs
        return obs


# ------------------------------ rollout -------------------------------------

class Runner:
    def __init__(self, game, model, device, auto_defend=True, overflow_cap=300):
        self.game, self.model, self.device = game, model, device
        self.auto_defend = auto_defend
        self.overflow_cap = overflow_cap
        self.results = collections.deque(maxlen=50)
        self.ep_returns = collections.deque(maxlen=50)
        self.ep_ret = 0.0
        self.episodes = 0
        self.state = F.encode(self.game.reset())

    def _tensors(self, state):
        P, G, sm, tm, _aux = state
        t = lambda x, dt: torch.as_tensor(x, dtype=dt, device=self.device).unsqueeze(0)
        return t(P, torch.float32), t(G, torch.float32), t(sm, torch.bool), t(tm, torch.bool)

    @torch.no_grad()
    def rollout(self, n_steps):
        B = dict(P=[], G=[], sm=[], tm=[], a=[], logp=[], val=[], rew=[], done=[])
        for _ in range(n_steps):
            P, G, sm, tm, aux = self.state
            tP, tG, tsm, ttm = self._tensors(self.state)
            ds, dt_, df, v = self.model.dists(tP, tG, tsm, ttm)
            a_s, a_t, a_f = ds.sample(), dt_.sample(), df.sample()
            logp = (ds.log_prob(a_s) + dt_.log_prob(a_t) + df.log_prob(a_f)).item()

            moves = F.decode_action(aux, int(a_s), int(a_t), int(a_f))
            if self.auto_defend:
                moves = moves + F.auto_defend(aux)
            if self.overflow_cap:
                moves = moves + F.overflow_attack(aux, cap=self.overflow_cap)
            obs, rew, done, outcome = self.game.step(moves)

            # 5. Comet overcommit penalty — sending ships to a comet about to expire
            rows = aux.get("rows", [])
            if moves and int(a_t) < len(rows):
                tgt = rows[int(a_t)]
                if tgt.get("comet", False):
                    pp = aux.get("comet_paths", {}).get(tgt["id"])
                    if pp is not None:
                        path, idx = pp
                        turns_left = len(path) - 1 - idx
                        if turns_left < 5:
                            src = rows[int(a_s)]
                            avail = int(src["ships"])
                            frac = F.FRACS[int(a_f)]
                            n = max(1, avail - 1) if frac >= 0.999 else max(1, int(frac * avail))
                            n = min(n, avail)
                            rew -= n * max(0.0, 1.0 - turns_left / 5.0) / 200.0

            self.ep_ret += rew

            B["P"].append(P); B["G"].append(G); B["sm"].append(sm); B["tm"].append(tm)
            B["a"].append((int(a_s), int(a_t), int(a_f)))
            B["logp"].append(logp); B["val"].append(v.item())
            B["rew"].append(rew); B["done"].append(float(done))

            if done:
                self.results.append(outcome)
                self.ep_returns.append(self.ep_ret)
                self.ep_ret = 0.0
                self.episodes += 1
                obs = self.game.reset()
            self.state = F.encode(obs)

        tP, tG, tsm, ttm = self._tensors(self.state)
        _, _, _, v_last = self.model.dists(tP, tG, tsm, ttm)
        return B, v_last.item()


def compute_gae(rew, val, done, v_last, gamma, lam):
    """Generalized Advantage Estimation (Schulman et al. 2016, arXiv:1506.02438)."""
    n = len(rew)
    adv = np.zeros(n, dtype=np.float32)
    last = 0.0
    for t in reversed(range(n)):
        next_v = v_last if t == n - 1 else val[t + 1]
        nonterminal = 1.0 - done[t]
        delta = rew[t] + gamma * next_v * nonterminal - val[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    returns = adv + np.asarray(val, dtype=np.float32)
    return adv, returns


# ------------------------------ PPO update ----------------------------------

def ppo_update(model, optimizer, B, adv, ret, args, device):
    N = len(B["rew"])
    P = torch.as_tensor(np.stack(B["P"]), dtype=torch.float32, device=device)
    G = torch.as_tensor(np.stack(B["G"]), dtype=torch.float32, device=device)
    SM = torch.as_tensor(np.stack(B["sm"]), dtype=torch.bool, device=device)
    TM = torch.as_tensor(np.stack(B["tm"]), dtype=torch.bool, device=device)
    A = torch.as_tensor(np.asarray(B["a"]), dtype=torch.long, device=device)
    LP = torch.as_tensor(np.asarray(B["logp"]), dtype=torch.float32, device=device)
    ADV = torch.as_tensor(adv, dtype=torch.float32, device=device)
    RET = torch.as_tensor(ret, dtype=torch.float32, device=device)
    ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)

    idx = np.arange(N)
    pl = vl = ent = 0.0
    for _ in range(args.epochs):
        np.random.shuffle(idx)
        for s in range(0, N, args.minibatch):
            b = torch.as_tensor(idx[s:s + args.minibatch], device=device)
            ds, dt_, df, v = model.dists(P[b], G[b], SM[b], TM[b])
            logp = ds.log_prob(A[b, 0]) + dt_.log_prob(A[b, 1]) + df.log_prob(A[b, 2])
            ratio = (logp - LP[b]).exp()
            surr1 = ratio * ADV[b]
            surr2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * ADV[b]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = ((v - RET[b]) ** 2).mean()
            entropy = (ds.entropy() + dt_.entropy() + df.entropy()).mean()
            loss = policy_loss + args.vcoef * value_loss - args.ecoef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            pl, vl, ent = policy_loss.item(), value_loss.item(), entropy.item()
    return pl, vl, ent


# ------------------------------ evaluation ----------------------------------

def evaluate(weights_path, opponent, n_games, players="mix", auto_defend=True, overflow_cap=300):
    """Play full greedy games using the exported numpy weights — i.e. the exact
    code path the Kaggle submission (main.py) uses."""
    policy = F.NumpyPolicy(np.load(weights_path))
    w = d = l = 0
    diffs = []
    for i in range(n_games):
        game = Game(opponent, seat=i, players=players)  # seat rotates (mod n)
        obs = game.reset()
        done = False
        outcome = 0
        while not done:
            moves = F.greedy_moves(policy, obs, repair=True,
                                   defend=auto_defend, overflow_cap=overflow_cap)
            obs, _r, done, outcome = game.step(moves)
        m, o = F.totals(obs)
        diffs.append(m - o)
        w += outcome == 1; d += outcome == 0; l += outcome == -1
        print(f"game {i + 1}/{n_games} ({game.num_players}p): "
              f"{'WIN' if outcome == 1 else 'LOSS' if outcome == -1 else 'DRAW'}"
              f"  my={m:.0f} best_opp={o:.0f}")
    print(f"\nvs {opponent}: {w} wins / {d} draws / {l} losses "
          f"(winrate {w / max(n_games, 1):.0%}, mean ship diff {np.mean(diffs):+.1f})")


def _eval_greedy(policy, opponent, n_games, players="mix",
                 auto_defend=True, overflow_cap=300):
    """Evaluate a NumpyPolicy against a fixed opponent (same as evaluate() but
    takes a policy object directly instead of a file path)."""
    w = d = l = 0
    diffs = []
    for i in range(n_games):
        game = Game(opponent, seat=i, players=players)
        obs = game.reset()
        done = False
        outcome = 0
        while not done:
            moves = F.greedy_moves(policy, obs, repair=True,
                                   defend=auto_defend, overflow_cap=overflow_cap)
            obs, _r, done, outcome = game.step(moves)
        m, o = F.totals(obs)
        diffs.append(m - o)
        w += outcome == 1; d += outcome == 0; l += outcome == -1
        print(f"game {i + 1}/{n_games} ({game.num_players}p): "
              f"{'WIN' if outcome == 1 else 'LOSS' if outcome == -1 else 'DRAW'}"
              f"  my={m:.0f} best_opp={o:.0f}")
    print(f"\nvs {opponent}: {w} wins / {d} draws / {l} losses "
          f"(winrate {w / max(n_games, 1):.0%}, mean ship diff {np.mean(diffs):+.1f})")


# ------------------------------ main -----------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="PPO for Orbit Wars")
    ap.add_argument("--total-steps", type=int, default=300_000)
    ap.add_argument("--rollout", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--vcoef", type=float, default=0.5)
    ap.add_argument("--ecoef", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--opponent", type=str, default="sniper.py",
                    help="opponent agent file (default: the repo's nearest-planet sniper)")
    ap.add_argument("--players", type=str, default="mix", choices=["2", "4", "mix"],
                    help="2 = 1v1, 4 = 1v3, mix = random per episode (default)")
    ap.add_argument("--no-auto-defend", action="store_true",
                    help="disable the scripted low-garrison reinforcement rule")
    ap.add_argument("--overflow-cap", type=int, default=300,
                    help="garrisons above this MUST attack (0 disables)")
    ap.add_argument("--ckpt", type=str, default="ppo_orbitwars.pt")
    ap.add_argument("--out", type=str, default="ppo_weights.npz",
                    help="numpy weights consumed by main.py")
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--eval", type=int, default=0, help="evaluate N games and exit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--selfplay", action="store_true",
                    help="train against greedy argmax of current policy (self-play)")
    ap.add_argument("--strong-agent", action="store_true",
                    help="train against the torch-based strong agent (sota_agents/strong_agent)")
    ap.add_argument("--snapshot-every", type=int, default=0,
                    help="save a self-play snapshot every N updates (0 = disabled)")
    ap.add_argument("--wandb", action="store_true",
                    help="enable Weights & Biases logging")
    ap.add_argument("--wandb-project", type=str, default="space-gladiators-orbit-wars",
                    help="wandb project name")
    ap.add_argument("--wandb-entity", type=str, default=None,
                    help="wandb entity/username")
    ap.add_argument("--wandb-tags", type=str, default=None,
                    help="comma-separated tags for the wandb run")
    return ap.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    players = int(args.players) if args.players in ("2", "4") else "mix"
    auto_defend = not args.no_auto_defend

    if args.selfplay and args.strong_agent:
        print("error: --selfplay and --strong-agent are mutually exclusive"); sys.exit(1)

    if args.eval:
        if args.selfplay:
            ckpt_path = args.resume if args.resume and os.path.exists(args.resume) else args.ckpt
            device = torch.device(args.device)
            model = TorchPolicy().to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.eval()
            policy = F.NumpyPolicy({k: v.detach().cpu().numpy()
                                    for k, v in model.state_dict().items()})
            _eval_greedy(policy, args.opponent, args.eval, players=players,
                         auto_defend=auto_defend, overflow_cap=args.overflow_cap)
        elif args.strong_agent:
            evaluate(args.out, "sota_agents/strong_agent/main.py", args.eval,
                     players=players, auto_defend=auto_defend,
                     overflow_cap=args.overflow_cap)
        else:
            evaluate(args.out, args.opponent, args.eval, players=players,
                     auto_defend=auto_defend, overflow_cap=args.overflow_cap)
        return

    device = torch.device(args.device)
    model = TorchPolicy().to(device)
    if args.resume and os.path.exists(args.resume):
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"resumed from {args.resume}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.selfplay:
        game = SelfPlayGame(model, device, players=players,
                            auto_defend=auto_defend, overflow_cap=args.overflow_cap)
        opp_label = "self (greedy)"
    elif args.strong_agent:
        game = StrongAgentGame(players=players)
        opp_label = "strong_agent"
    else:
        game = Game(args.opponent, players=players)
        opp_label = args.opponent

    runner = Runner(game, model, device,
                    auto_defend=auto_defend, overflow_cap=args.overflow_cap)
    steps, update = 0, 0
    t0 = time.time()
    print(f"training on {device} vs '{opp_label}' | players={args.players} "
          f"| auto_defend={auto_defend} "
          f"({sum(p.numel() for p in model.parameters()):,} params)")

    if args.wandb:
        if not _HAS_WANDB:
            print("wandb not installed; install with `pip install wandb`")
        else:
            tags = args.wandb_tags.split(",") if args.wandb_tags else None
            _wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                tags=tags,
                config={
                    "total_steps": args.total_steps,
                    "rollout": args.rollout,
                    "epochs": args.epochs,
                    "minibatch": args.minibatch,
                    "lr": args.lr,
                    "gamma": args.gamma,
                    "lam": args.lam,
                    "clip": args.clip,
                    "vcoef": args.vcoef,
                    "ecoef": args.ecoef,
                    "max_grad_norm": args.max_grad_norm,
                    "players": args.players,
                    "selfplay": args.selfplay,
                    "strong_agent": args.strong_agent,
                    "opponent": opp_label,
                    "auto_defend": auto_defend,
                    "overflow_cap": args.overflow_cap,
                    "seed": args.seed,
                    "device": str(device),
                    "params": sum(p.numel() for p in model.parameters()),
                    "policy_arch": "v4",
                },
            )
            _wandb.watch(model, log="gradients", log_freq=100)

    while steps < args.total_steps:
        B, v_last = runner.rollout(args.rollout)
        adv, ret = compute_gae(B["rew"], B["val"], B["done"], v_last, args.gamma, args.lam)
        pl, vl, ent = ppo_update(model, optimizer, B, adv, ret, args, device)
        steps += args.rollout
        update += 1

        # save checkpoint (torch) + exported weights (numpy, used by main.py)
        torch.save(model.state_dict(), args.ckpt)
        np.savez(args.out, **{k: v.detach().cpu().numpy() for k, v in model.state_dict().items()})

        # optional snapshot for self-play
        if args.selfplay and args.snapshot_every and update % args.snapshot_every == 0:
            snap_path = f"selfplay_upd{update}.pt"
            torch.save(model.state_dict(), snap_path)

        res = list(runner.results)
        winrate = (sum(1 for r in res if r == 1) / len(res)) if res else float("nan")
        mean_ret = np.mean(runner.ep_returns) if runner.ep_returns else float("nan")
        sps = steps / (time.time() - t0)
        print(f"upd {update:4d} | steps {steps:>8,} | eps {runner.episodes:4d} "
              f"| winrate(last{len(res)}) {winrate:5.0%} | ep_ret {mean_ret:+7.2f} "
              f"| pi {pl:+.3f} v {vl:.3f} ent {ent:.2f} | {sps:,.0f} steps/s")

        if args.wandb and _wandb.run is not None:
            wins = sum(1 for r in res if r == 1)
            draws = sum(1 for r in res if r == 0)
            losses = sum(1 for r in res if r == -1)
            _wandb.log({
                "update": update,
                "steps": steps,
                "episodes": runner.episodes,
                "winrate": winrate,
                "mean_ep_return": mean_ret,
                "policy_loss": pl,
                "value_loss": vl,
                "entropy": ent,
                "steps_per_sec": sps,
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "n_episodes_in_window": len(res),
            })

    if args.wandb and _wandb.run is not None:
        _wandb.log_artifact(args.ckpt, type="model", name="ppo_checkpoint")
        _wandb.finish()
    print(f"done. saved {args.ckpt} and {args.out}")


if __name__ == "__main__":
    main()
