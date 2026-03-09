#!/bin/zsh
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
cd "${script_dir}/.."

run_tag="${1:-overnight-$(date +%Y%m%d-%H%M%S)}"
duration_hours="${2:-8}"
shift $(( $# >= 2 ? 2 : $# ))

mkdir -p results/overnight

launcher_log="results/overnight/${run_tag}.launcher.log"
pid_file="results/overnight/${run_tag}.pid"

./tools/detach_exec.py "$pid_file" "$launcher_log" \
  ./.venv/bin/python -u ./tools/overnight_mlx.py \
  --run-tag "$run_tag" \
  --duration-hours "$duration_hours" \
  "$@"

pid="$(cat "$pid_file")"

printf 'run_tag: %s\n' "$run_tag"
printf 'pid: %s\n' "$pid"
printf 'launcher_log: %s\n' "$launcher_log"
printf 'pid_file: %s\n' "$pid_file"
