# TODO

## Phase 1 — Deterministic Path System
- Path calculator module: given source & target, return angle + whether it reaches
- Account for orbiting planets (predict intercept, not just current position)
- Sun and out-of-bounds collision checks

## Phase 2 — Elemental Tasks
- Capture — send enough ships to take a target
- Reinforce — beef up a friendly planet
- SendShips — raw send (base primitive)
- Wait — accumulate / skip

## Phase 3 — Near-to-Far Strategy
- Nearest-neighbor greedy: each planet attacks closest unowned target
- Perimeter expansion: after taking neighbors, target planets just beyond owned territory
- Phase awareness: early → take neutrals, mid → attack borders, late → push & consolidate

## Phase 4 — Agent Integration
- Agent state class: parse observation, track owned planets, fleets, angular velocity
- Strategy entry point: swap strategies easily
- Clean main.py: wire state + strategy + tasks into kaggle-compatible agent function
