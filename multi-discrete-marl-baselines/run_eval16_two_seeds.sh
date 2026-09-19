#!/usr/bin/env bash
set -euo pipefail

TOWER="${1:?Usage: $0 <5|10|15|20> <algorithm>}"
ALGO="${2:?Usage: $0 <5|10|15|20> <algorithm>}"

case "$TOWER" in
  5|10)
    EPISODE_LENGTH=200
    NUM_ENV_STEPS=402000
    ;;
  15)
    EPISODE_LENGTH=200
    NUM_ENV_STEPS=602000
    ;;
  20)
    EPISODE_LENGTH=300
    NUM_ENV_STEPS=902000
    ;;
  *) echo "Unsupported tower count: $TOWER"; exit 1 ;;
esac

case "$ALGO" in
  mappo|rmappo|happo|hatrpo|rode|roma|mat|mat_dec) ;;
  *) echo "Unsupported algorithm: $ALGO"; exit 1 ;;
esac

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${ANTIATTACK_PYTHON:-python}"
TRAIN="$PROJECT/onpolicy/scripts/train/train_antiattack_brief.py"
CONFIG="$PROJECT/onpolicy/envs/anti_Attack/configs/env_config_${TOWER}tower.json5"
EXPERIMENT="brief_${TOWER}tower_${ALGO}_wd1e4_1xc_eval16_evaldetail"
SEEDS=(2032 2033)

test -x "$PYTHON" || { echo "Python not executable: $PYTHON"; exit 1; }
test -f "$TRAIN" || { echo "Training entry not found: $TRAIN"; exit 1; }
test -f "$CONFIG" || { echo "Environment config not found: $CONFIG"; exit 1; }

cd "$PROJECT"
export PYTHONPATH="$PROJECT"
export MPLBACKEND=Agg
export ANTIATTACK_ENV_CONFIG="$CONFIG"
mkdir -p server_logs

for seed in "${SEEDS[@]}"; do
  echo "===== tower=$TOWER, algorithm=$ALGO, seed=$seed, eval_interval=16 ====="

  "$PYTHON" "$TRAIN" \
    --env_name towerDefense \
    --algorithm_name "$ALGO" \
    --experiment_name "$EXPERIMENT" \
    --lr 1e-5 \
    --weight_decay 1e-4 \
    --num_env_steps "$NUM_ENV_STEPS" \
    --n_rollout_threads 1 \
    --episode_length "$EPISODE_LENGTH" \
    --seed "$seed" \
    --n_eval_rollout_threads 32 \
    --eval_interval 16 \
    > "server_logs/${ALGO}_${TOWER}tower_eval16_seed${seed}.log" 2>&1
done

echo "All runs finished: tower=$TOWER, algorithm=$ALGO"
