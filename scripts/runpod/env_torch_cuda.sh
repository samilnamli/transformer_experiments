#!/usr/bin/env bash
# Source from pipeline scripts so torch loads bundled NCCL/CUDA libs first.
set -euo pipefail
TORCH_LIB="$(uv run python -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null || true)"
if [[ -n "${TORCH_LIB}" && -d "${TORCH_LIB}" ]]; then
  export LD_LIBRARY_PATH="${TORCH_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
