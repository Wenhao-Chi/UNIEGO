#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

cd "$(dirname "$0")/../../.."

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_DIR="${CONFIG_DIR:-exp/proxy_distill/assembly101}"
EXTRA_OPTS=("$@")

run_cmd() {
  printf '\n>>>'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

run_configs() {
  local phase="$1"
  shift
  local configs=("$@")

  if [[ "${#configs[@]}" == "0" ]]; then
    printf 'No configs found for %s\n' "$phase" >&2
    exit 1
  fi

  printf '\n[%s]\n' "$phase"
  for cfg in "${configs[@]}"; do
    run_cmd "$PYTHON_BIN" tools/run_net.py --cfg "$cfg" "${EXTRA_OPTS[@]}"
  done
}

printf 'Config dir: %s\n' "$CONFIG_DIR"

if [[ "${RUN_STAGE1:-1}" == "1" ]]; then
  run_configs "stage1 proxy training" "$CONFIG_DIR"/stage1_*.yaml
fi

if [[ "${RUN_GEN1_INFER:-1}" == "1" ]]; then
  run_configs "stage1 proxy inference" "$CONFIG_DIR"/infer_gen1_*.yaml
fi

if [[ "${RUN_MERGE:-1}" == "1" ]]; then
  printf '\n[model merging]\n'
  run_cmd "$PYTHON_BIN" tools/merging_models.py \
    --cfg "$CONFIG_DIR/merge_stage1.yaml" "${EXTRA_OPTS[@]}"
fi

if [[ "${RUN_STAGE2:-1}" == "1" ]]; then
  run_configs "stage2 proxy training" "$CONFIG_DIR/stage2_dist.yaml"
fi

if [[ "${RUN_TEST_STAGE2:-0}" == "1" ]]; then
  run_configs "stage2 proxy testing" "$CONFIG_DIR/test_stage2.yaml"
fi

printf '\nAssembly101 proxy pipeline complete.\n'
