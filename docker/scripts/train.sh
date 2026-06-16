#!/bin/bash 
# Convenience launcher for dockerized PPO training.
# Usage: ./docker/scripts/train.sh [extra args for train_ppo.py]
#
# Examples:
#   ./docker/scripts/train.sh                                          # 3M steps vs sniper + wandb
#   ./docker/scripts/train.sh --total-steps 1000000 --no-auto-defend   # custom
#   ./docker/scripts/train.sh --resume ppo_orbitwars.pt                # resume
#   ./docker/scripts/train.sh -- --strong-agent                        # train vs strong agent
#   WANDB_MODE=offline ./docker/scripts/train.sh                       # offline logging

cd "$(dirname "$0")/../.."

export WANDB_MODE="${WANDB_MODE:-online}"

exec docker compose -f docker/docker-compose.yml run --rm train "$@"
