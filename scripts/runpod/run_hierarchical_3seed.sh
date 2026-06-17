#!/usr/bin/env bash
# HIT-ASR on VoxPopuli: hierarchical_transformer, 3 seeds, batch 192.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# shellcheck source=/dev/null
source "$ROOT/scripts/runpod/env_torch_cuda.sh"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== HIT-ASR 3-seed (42, 0, 1337) batch 192 ==="
uv run python run.py experiment=voxpopuli_hierarchical_3seed_runpod "$@"
