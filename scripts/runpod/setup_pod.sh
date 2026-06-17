#!/usr/bin/env bash
# THE ONLY setup command for RunPod. Do not run bare `uv sync` on the pod.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export UV_LINK_MODE=copy

echo "=== RunPod setup (Python 3.10 + torch 2.6.0+cu124) ==="
nvidia-smi || { echo "ERROR: no GPU visible"; exit 1; }

rm -rf .venv
uv venv --python 3.10

# Install all deps EXCEPT torch — uv sync would pull PyPI torch 2.11 + CUDA 13.
uv sync --no-install-package torch

# Self-contained GPU wheel (no PyPI cuda-toolkit / nvidia-*-cu13 stack).
uv pip install --reinstall --no-deps "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124

# shellcheck source=/dev/null
source "$ROOT/scripts/runpod/env_torch_cuda.sh"

uv run python - <<'PY'
import torch

assert torch.cuda.is_available(), "CUDA not available — check template + setup_pod.sh"
name = torch.cuda.get_device_name(0)
print(f"OK: torch {torch.__version__}, device={name}")
PY

echo "=== Pod setup complete. Next: bash scripts/runpod/tmux_voxpopuli.sh ==="
