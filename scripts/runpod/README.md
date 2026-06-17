# RunPod — VoxPopuli paper-ready experiments

## Recommended GPU

| GPU | Price | VRAM | RAM | Verdict |
|-----|-------|------|-----|---------|
| **L40S** | ~$0.86/hr | 48 GB | 125 GB | **Current setup** — `batch_size=384` in runpod config |
| **RTX 4090** | ~$0.69/hr | 24 GB | 61 GB | Use `data.batch_size=128` override if you switch GPUs |
| RTX 3090 | ~$0.46/hr | 24 GB | 125 GB | Cheaper; override `data.batch_size=128` |
| L4 | ~$0.39/hr | 24 GB | 50 GB | Budget; may be tight on RAM for eager-load |

**Estimated wall time on L40S (batch 384):** ~1 h Whisper re-decode + ~4–7 h full main results (16 seeds × 10 methods).

If training OOMs, re-run step 3 with `data.batch_size=256` or `128`.

## One-time pod setup

```bash
git clone https://github.com/samilnamli/transformer_experiments.git
cd transformer_experiments
git checkout runpod/voxpopuli-main-results

# Copy the bundled parquet (24 GB) — choose one:
#   A) From your laptop (if you have it locally):
#      rsync -avP /path/to/combined_features_with_transcripts.parquet \
#        root@<pod-ip>:/workspace/s2t-tr-dev/configs/data/processed/facebook_voxpopuli/
#   B) Google Drive fallback (slower):
#      uv run python -m src.data.prepare --dataset voxpopuli

make create_environment
uv sync
# RunPod PyTorch images usually ship CUDA torch; if not:
# uv pip install torch --index-url https://download.pytorch.org/whl/cu124

uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Optional: set DagsHub MLflow env vars (`DAGSHUB_USER_TOKEN`, `DAGSHUB_TRACKING_URI`) before experiments.

## Run everything in tmux (detach-safe)

```bash
bash scripts/runpod/tmux_voxpopuli.sh
# attach later:
tmux attach -t voxpopuli
```

Logs land in `logs/runpod/voxpopuli_<timestamp>.log`.

## Manual step-by-step

```bash
# 1) Fix Whisper WER (decoder re-decode from cached features)
uv run python scripts/redecode_voxpopuli_whisper_overrides_from_features.py

# 2) Sanity check
uv run python scripts/check_voxpopuli.py

# 3) Paper-ready main results (16 seeds, stats tests)
uv run python run.py experiment=main_results_voxpopuli_runpod
```

Results are logged to MLflow (`results/test_wer_comparison.json` on the parent run).
