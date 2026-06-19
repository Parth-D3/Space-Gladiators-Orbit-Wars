"""Learnable scoring network for the strong_agent planner.

Replaces the hand-crafted ``competitive_score`` (``me - sum(opp)``) with a
tiny MLP trained via REINFORCE self-play.  The network consumes per-candidate
features (per-player net-ship delta + auxiliary context) and emits a scalar
score; higher = more promising launch.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ScoreNetwork(nn.Module):
    """Tiny MLP scorer.

    Architecture::

        [10 features] → Linear(10, 32) → ReLU → Linear(32, 32) → ReLU → Linear(32, 1)

    Features per candidate
    ----------------------
    *net_delta[0..3]* (4, padded to 4 players) — per-player net ship change
    *ships_sent* (1)   — number of ships this launch commits
    *eta* (1)          — turns until arrival
    *target_prod* (1)  — production rate of the target planet
    *source_prod* (1)  — production rate of the source planet
    *is_owned* (1)     — target currently owned by us
    *step_frac* (1)    — current step / total steps

    Total: **10** floats.
    """

    MAX_PLAYERS: int = 4
    N_AUX: int = 6  # ships_sent, eta, tgt_prod, src_prod, is_owned, step_frac

    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        in_dim: int = self.MAX_PLAYERS + self.N_AUX
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Score each candidate.  Input ``[*, in_dim]`` → output ``[*]``."""
        return self.net(x).squeeze(-1)

    # ------------------------------------------------------------------
    # Feature building
    # ------------------------------------------------------------------

    @staticmethod
    def build_features(
        net_delta: Tensor,      # [*, A]  (A = 2 or 4)
        ships_sent: Tensor,     # [*]
        eta: Tensor,            # [*]
        target_prod: Tensor,    # [*]
        source_prod: Tensor,    # [*]
        is_owned: Tensor,       # [*]  bool
        step_frac: Tensor,      # [*]
    ) -> Tensor:
        """Stack raw quantities into a flat feature vector ``[*, MAX_PLAYERS + N_AUX]``."""
        A = net_delta.shape[-1]
        if A < ScoreNetwork.MAX_PLAYERS:
            pad = torch.zeros(
                *net_delta.shape[:-1],
                ScoreNetwork.MAX_PLAYERS - A,
                device=net_delta.device,
                dtype=net_delta.dtype,
            )
            net_delta = torch.cat([net_delta, pad], dim=-1)
        aux = torch.stack(
            [
                ships_sent,
                eta,
                target_prod,
                source_prod,
                is_owned.float(),
                step_frac,
            ],
            dim=-1,
        )
        return torch.cat([net_delta, aux], dim=-1)

    # ------------------------------------------------------------------
    # High-level scoring helper used inside the planner pipeline
    # ------------------------------------------------------------------

    def score_candidates(
        self,
        diff,                     # GarrisonFlowDiff
        launches,                 # LaunchSet
        prod: Tensor,
        status,                   # PlanetGarrisonStatus
        step_frac: float,
        player_id: int,
    ) -> Tensor:
        """Build features from planner internals and produce scores.

        This is the single call-site hook inside ``score_candidates`` in
        ``planner_core.py``.  Every tensor is detached before entering the
        network to avoid back-propagating through the game engine.
        """
        src = launches.source_slots                     # [C, L]  (L=1 typically)
        tgt = launches.target_slots
        P = prod.shape[0]
        src_safe = src.clamp(0, P - 1)
        tgt_safe = tgt.clamp(0, P - 1)

        net_delta = diff.net_ship_delta.detach()        # [C, A]
        ships_sent = launches.ships.detach()            # [C, L]
        eta = launches.eta.detach()                     # [C, L]

        init_owner = status.owner[:, 0]                 # [P]
        tgt_prod = prod[tgt_safe].detach()              # [C, L]
        src_prod = prod[src_safe].detach()              # [C, L]
        is_owned = (init_owner[tgt_safe] == player_id).detach()  # [C, L]

        # Average over the L axis if multiple launches per candidate.
        valid = launches.valid.detach()                 # [C, L]
        w = valid.float()                               # [C, L]
        w_sum = w.sum(dim=-1, keepdim=True).clamp(min=1.0)
        avg = lambda t: (t * w).sum(dim=-1) / w_sum.squeeze(-1)

        feats = self.build_features(
            net_delta=net_delta,                        # [C, A]
            ships_sent=avg(ships_sent),                 # [C]
            eta=avg(eta),                               # [C]
            target_prod=avg(tgt_prod),                  # [C]
            source_prod=avg(src_prod),                  # [C]
            is_owned=avg(is_owned),                     # [C]
            step_frac=torch.full(
                (net_delta.shape[0],), step_frac,
                device=net_delta.device, dtype=net_delta.dtype,
            ),
        )
        return self(feats)                              # [C]
