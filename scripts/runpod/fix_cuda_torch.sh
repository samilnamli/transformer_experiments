#!/usr/bin/env bash
# RunPod: uv sync installs PyPI torch 2.11 + split CUDA/NCCL packages that conflict
# with GPU wheels (undefined symbol ncclCommWindowDeregister, driver mismatch, etc.).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

echo "=== RunPod torch / CUDA fix ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "WARNING: nvidia-smi not found"
fi

purge_torch_cuda_stack() {
  echo "Purging torch / CUDA / NCCL packages from venv..."
  # Repeat until nothing left — uv may leave transitive nvidia-* packages behind.
  for _ in 1 2 3; do
    mapfile -t pkgs < <(uv pip freeze 2>/dev/null | grep -iE '^(torch|triton|cuda-|nvidia-)' | cut -d= -f1 || true)
    if ((${#pkgs[@]} == 0)); then
      break
    fi
    uv pip uninstall "${pkgs[@]}" 2>/dev/null || true
  done
}

purge_torch_cuda_stack

TORCH_INDEX="https://download.pytorch.org/whl/cu124"
# Driver 570 + CUDA 12.8: use self-contained cu124 wheels (not PyPI torch 2.11 / cu128).
CANDIDATES=(2.6.0 2.5.1 2.4.1)

for ver in "${CANDIDATES[@]}"; do
  echo "--- Trying torch==${ver} (${TORCH_INDEX}, --no-deps) ---"
  purge_torch_cuda_stack
  if ! uv pip install --reinstall --no-deps "torch==${ver}" --index-url "${TORCH_INDEX}"; then
    continue
  fi
  # Non-CUDA runtime deps only (never install nvidia-* / cuda-toolkit from PyPI).
  uv pip install sympy filelock typing-extensions networkx jinja2 fsspec 2>/dev/null || true

  if uv run python - <<'PY'
import os
import torch

# Prefer torch's bundled CUDA/NCCL libs over any host copies.
torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
os.environ["LD_LIBRARY_PATH"] = torch_lib + (
    ":" + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else ""
)
import importlib
importlib.reload(torch)
if not torch.cuda.is_available():
    raise SystemExit(1)
print(f"OK: torch {torch.__version__}, device={torch.cuda.get_device_name(0)}")
PY
  then
    echo "NOTE: Do not run 'uv sync' without re-running this script."
    echo "If imports fail in other shells, run: export LD_LIBRARY_PATH=\$(uv run python -c \"import os,torch; print(os.path.join(os.path.dirname(torch.__file__),'lib'))\"):\$LD_LIBRARY_PATH"
    exit 0
  fi
  echo "torch ${ver} failed CUDA check; trying older build..."
done

cat <<'EOF' >&2
FAILED: no compatible torch wheel found.

Manual recovery on this pod (driver CUDA 12.8):
  purge_torch_cuda_stack  # or re-run this script
  uv pip install --reinstall --no-deps torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
  uv pip install sympy filelock typing-extensions networkx jinja2 fsspec
  uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

If still broken, recreate the venv:
  rm -rf .venv && make create_environment && uv sync && bash scripts/runpod/fix_cuda_torch.sh
EOF
exit 1
