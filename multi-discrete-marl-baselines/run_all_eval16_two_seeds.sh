#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAX_JOBS="${MAX_JOBS:-4}"
TOWERS=(5 10 15 20)
ALGORITHMS=(mappo rmappo happo hatrpo rode roma mat mat_dec)

mkdir -p "$PROJECT/server_logs"

running_jobs() {
  jobs -pr | wc -l
}

for tower in "${TOWERS[@]}"; do
  for algorithm in "${ALGORITHMS[@]}"; do
    while (( $(running_jobs) >= MAX_JOBS )); do
      wait -n
    done

    echo "Starting tower=$tower algorithm=$algorithm"
    "$PROJECT/run_eval16_two_seeds.sh" "$tower" "$algorithm" \
      > "$PROJECT/server_logs/${algorithm}_${tower}tower_eval16_launcher.log" 2>&1 &
  done
done

wait
echo "All 5/10/15/20-tower eval16 jobs finished."
