#!/usr/bin/env bash
# RunPod: uv sync installs PyPI torch 2.11 + cuda-toolkit (CUDA 13), which does not
# match typical RunPod drivers. Replace with a self-contained PyTorch cu124 wheel.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

echo "=== RunPod torch / CUDA fix ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "WARNING: nvidia-smi not found"
fi

echo "Removing PyPI torch + split CUDA packages (they conflict with GPU wheels)..."
uv pip uninstall \
  torch triton cuda-toolkit cuda-bindings \
  nvidia-cudnn-cu13 nvidia-cusparselt-cu13 nvidia-nccl-cu13 nvidia-nvshmem-cu13 \
  nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 \
  nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 \
  nvidia-cufile-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 \
  nvidia-cusparse-cu12 nvidia-cusparselt-cu12 nvidia-nccl-cu12 \
  nvidia-nvjitlink-cu12 nvidia-nvshmem-cu12 nvidia-nvtx-cu12 \
  2>/dev/null || true

TORCH_INDEX="https://download.pytorch.org/whl/cu124"
# Newest first; torch 2.11+cu128 needs a driver newer than most RunPod hosts (12.8).
CANDIDATES=(2.6.0 2.5.1 2.4.1)

for ver in "${CANDIDATES[@]}"; do
  echo "--- Trying torch==${ver} (${TORCH_INDEX}) ---"
  uv pip install --reinstall "torch==${ver}" --index-url "${TORCH_INDEX}" || continue
  if uv run python - <<'PY'
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
  then
    uv run python - <<'PY'
import torch
print(f"OK: torch {torch.__version__}, device={torch.cuda.get_device_name(0)}")
PY
    echo "NOTE: Do not run 'uv sync' again without re-running this script."
    exit 0
  fi
  echo "torch ${ver} installed but CUDA unavailable; trying older build..."
  uv pip uninstall torch 2>/dev/null || true
done

cat <<'EOF' >&2
FAILED: no compatible torch wheel found.

Try on the pod:
  1) nvidia-smi   # confirm GPU + driver CUDA version
  2) Use RunPod template "PyTorch 2.2.0" (CUDA 12.1) and re-run this script
  3) Or manually: uv pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
EOF
exit 1
