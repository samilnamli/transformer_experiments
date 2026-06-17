#!/usr/bin/env bash
# Full VoxPopuli pipeline: Whisper fix → sanity check → paper-ready main results.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

mkdir -p logs/runpod

echo "=== [$(date -Is)] VoxPopuli RunPod pipeline ==="
echo "Repo: $ROOT"
echo "CUDA: $(uv run python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")' 2>/dev/null || echo UNAVAILABLE — run: bash scripts/runpod/fix_cuda_torch.sh)"
echo "Config: experiment=main_results_voxpopuli_runpod (data.batch_size=384 for 48 GB GPUs)"

if ! uv run python -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "ERROR: CUDA not available. Run: bash scripts/runpod/fix_cuda_torch.sh"
  exit 1
fi

PARQUET="configs/data/processed/facebook_voxpopuli/combined_features_with_transcripts.parquet"
if [[ ! -f "$PARQUET" ]]; then
  echo "Parquet missing at $PARQUET — trying prepare_data download..."
  uv run python -m src.data.prepare --dataset voxpopuli
fi

echo "=== [$(date -Is)] Step 1/3: Whisper re-decode (fixed language + anti-repetition) ==="
uv run python scripts/redecode_voxpopuli_whisper_overrides_from_features.py \
  --input-parquet "$PARQUET" \
  --output-npz data/processed/facebook_voxpopuli/whisper_decode_overrides_from_features.npz

echo "=== [$(date -Is)] Step 2/3: Sanity check ==="
uv run python scripts/check_voxpopuli.py

echo "=== [$(date -Is)] Step 3/3: main_results_voxpopuli_runpod (16 seeds) ==="
uv run python run.py experiment=main_results_voxpopuli_runpod

echo "=== [$(date -Is)] DONE ==="
