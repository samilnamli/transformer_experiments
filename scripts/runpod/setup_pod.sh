#!/usr/bin/env bash
# One-time RunPod environment setup (call after git clone).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

make create_environment
uv sync
bash scripts/runpod/fix_cuda_torch.sh

echo "=== Pod setup complete ==="
