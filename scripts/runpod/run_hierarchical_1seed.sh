#!/usr/bin/env bash
# Quick HIT-ASR on VoxPopuli: hierarchical_transformer, seed 42 only.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# shellcheck source=/dev/null
source "$ROOT/scripts/runpod/env_torch_cuda.sh"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== HIT-ASR 1-seed (batch 192) ==="
uv run python run.py experiment=voxpopuli_hierarchical_1seed_runpod "$@"
