#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

PYTHON_BIN="${PYTHON_BIN:-python}"
exec "$PYTHON_BIN" exp/proxy_distill/run_pipeline.py --dataset egoexo4d "$@"
