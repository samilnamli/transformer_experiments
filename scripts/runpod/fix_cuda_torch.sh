#!/usr/bin/env bash
# Emergency repair if someone ran bare `uv sync` and broke CUDA.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export UV_LINK_MODE=copy

echo "=== Repairing broken torch (PyPI CUDA 13 → cu124 wheel) ==="

purge() {
  mapfile -t pkgs < <(uv pip freeze 2>/dev/null | grep -iE '^(torch|triton|cuda-|nvidia-)' | cut -d= -f1 || true)
  ((${#pkgs[@]})) && uv pip uninstall "${pkgs[@]}" 2>/dev/null || true
}

purge
uv pip install --reinstall --no-deps "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124

# shellcheck source=/dev/null
source "$ROOT/scripts/runpod/env_torch_cuda.sh"

uv run python -c "import torch; assert torch.cuda.is_available(); print('OK:', torch.__version__, torch.cuda.get_device_name(0))"
