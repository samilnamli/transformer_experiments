# RunPod — start here

## Create the pod (do this exactly)

| Setting | Value |
|---------|--------|
| GPU | **L40S** (48 GB) |
| Template | **Runpod PyTorch 2.2.0** (`py3.10-cuda12.1.1`) |
| Disk | **≥ 80 GB** (parquet is 24 GB) |
| Volume | `/workspace` |

Do **not** use a bare Ubuntu template. Use an official **Runpod PyTorch** image.

## Setup (one command — never bare `uv sync`)

```bash
git clone https://github.com/samilnamli/transformer_experiments.git
cd transformer_experiments
git checkout runpod/voxpopuli-main-results

# parquet → configs/data/processed/facebook_voxpopuli/combined_features_with_transcripts.parquet

bash scripts/runpod/setup_pod.sh
```

`setup_pod.sh` installs deps **without** PyPI torch 2.11, then adds `torch==2.6.0+cu124`.

If you already ran `uv sync` and broke CUDA:

```bash
bash scripts/runpod/fix_cuda_torch.sh
```

## Run experiments

```bash
bash scripts/runpod/tmux_voxpopuli.sh
tmux attach -t voxpopuli
```

## Why previous setup failed

1. `uv sync` installed **torch 2.11 from PyPI** with **CUDA 13** (`cuda-toolkit`, `nvidia-*-cu13`).
2. RunPod driver supports **CUDA 12.8** — incompatible with that stack.
3. Patching afterward fought leftover NCCL/CUDA packages.

**Fix:** `pyproject.toml` now pins **linux → torch 2.6.0 from cu124 index**; `setup_pod.sh` skips torch during sync.

## Recommended GPU

| GPU | Verdict |
|-----|---------|
| **L40S** | Current — `batch_size=384` in runpod config |
| RTX 4090 | Override `data.batch_size=128` |

**Estimated wall time (L40S, batch 384):** ~1 h Whisper re-decode + ~4–7 h main results.

Optional: `DAGSHUB_USER_TOKEN`, `DAGSHUB_TRACKING_URI` for remote MLflow.

Results: `mlruns/` → `results/test_wer_comparison.json` on parent run.
