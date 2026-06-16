#!/bin/bash
# Convenience launcher for dockerized PPO training.
# Usage: ./docker/scripts/train.sh [args for train_ppo.py]
#
# Examples:
#   ./docker/scripts/train.sh                                                        # 3M steps vs sniper + wandb
#   ./docker/scripts/train.sh --selfplay --snapshot-every 500 --eval-past 20 --wandb  # self-play + past self eval
#   ./docker/scripts/train.sh --total-steps 10000000 --selfplay --wandb               # 10M steps self-play
#   ./docker/scripts/train.sh --resume ppo_orbitwars.pt                               # resume
#   WANDB_MODE=offline ./docker/scripts/train.sh --selfplay --wandb                   # offline logging

cd "$(dirname "$0")/../.."

export WANDB_MODE="${WANDB_MODE:-online}"

# Auto-detect the right docker-compose service based on training mode flags
SERVICE="train"
for arg in "$@"; do
  case "$arg" in
    --selfplay) SERVICE="train-selfplay"; break ;;
    --strong-agent) SERVICE="train-strong-agent"; break ;;
  esac
done

exec docker compose -f docker/docker-compose.yml run --rm "$SERVICE" "$@"
