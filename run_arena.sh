#!/usr/bin/env bash
set -euo pipefail

python train_ppo.py \
  --resume ppo_orbitwars.pt \
  --total-steps 10000 \
  --selfplay \
  --arena-every 50 \
  --arena-games 20 \
  --wandb \
  --wandb-project space-gladiators-orbit-wars \
  --wandb-tags self-play,arena
