#!/usr/bin/env bash
# RunPod ships CUDA 12.8 drivers; default PyPI torch 2.11 bundles CUDA 13 libs.
# Reinstall torch from the cu128 wheel index so CUDA initializes correctly.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

echo "=== Fixing torch for RunPod (cu128 wheel, keeps torch 2.11) ==="

uv pip uninstall torch nvidia-cudnn-cu13 nvidia-cusparselt-cu13 nvidia-nccl-cu13 nvidia-nvshmem-cu13 2>/dev/null || true

uv pip install "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128

uv run python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA still unavailable after cu128 reinstall. "
        "Check nvidia-smi and try the PyTorch 2.5 RunPod template."
    )
print(f"OK: torch {torch.__version__}, device={torch.cuda.get_device_name(0)}")
PY
