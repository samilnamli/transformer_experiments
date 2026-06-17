"""
Re-decode Whisper transcriptions (and recompute whisper WER) for the
bundled VoxPopuli legacy parquet, using stored ``whisper_features`` as
encoder outputs for Whisper's decoder.

Reads parquet **by row group** (not per-clip ``Dataset.__getitem__``) so
GPU decode is not bottlenecked on repeated 700 MB row-group decodes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import jiwer
import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from transformers.modeling_outputs import BaseModelOutput

from src.data.voxpopuli import resolve_voxpopuli_parquet


def _wer(reference: str, hypothesis: str) -> float:
    ref = (reference or "").strip().lower()
    hyp = (hypothesis or "").strip().lower()
    if not ref:
        return 1.0 if hyp else 0.0
    return float(jiwer.wer(ref, hyp))


def _features_to_tensor(feat_py: list, *, dtype: torch.dtype) -> torch.Tensor:
    arr = np.asarray(feat_py, dtype=np.float32)
    return torch.from_numpy(arr).to(dtype=dtype)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-parquet",
        type=str,
        default=str(resolve_voxpopuli_parquet()),
    )
    parser.add_argument(
        "--output-npz",
        type=str,
        default="data/processed/facebook_voxpopuli/whisper_decode_overrides_from_features.npz",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    args = parser.parse_args()

    parquet_path = Path(args.input_parquet)
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Device: {device} ({dtype})")

    meta = pq.read_metadata(parquet_path)
    n_total = meta.num_rows
    n = min(n_total, int(args.limit)) if args.limit is not None else n_total

    whisper_wer = np.zeros((n_total,), dtype=np.float32)
    whisper_transcription: List[str] = [""] * n_total

    processor = AutoProcessor.from_pretrained("openai/whisper-base")
    model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-base", dtype=dtype)
    model.to(device)
    model.eval()
    forced = processor.get_decoder_prompt_ids(language="en", task="transcribe")

    pf = pq.ParquetFile(parquet_path)
    global_offset = 0
    decoded = 0

    for rg_idx in range(pf.num_row_groups):
        if decoded >= n:
            break
        table = pf.read_row_group(
            rg_idx, columns=["ground_truth", "whisper_features"]
        )
        rg_n = table.num_rows
        gts = table["ground_truth"].to_pylist()
        feats = table["whisper_features"]

        for row in range(rg_n):
            idx = global_offset + row
            if idx >= n:
                break

            gt = gts[row]
            enc = _features_to_tensor(feats[row].as_py(), dtype=model.dtype).unsqueeze(0).to(device)
            T = enc.shape[1]
            encoder_outputs = BaseModelOutput(last_hidden_state=enc)

            with torch.no_grad():
                token_ids = model.generate(
                    encoder_outputs=encoder_outputs,
                    attention_mask=torch.ones((1, T), device=device, dtype=torch.long),
                    forced_decoder_ids=forced,
                    max_new_tokens=args.max_new_tokens,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                )
            hyp = processor.batch_decode(token_ids, skip_special_tokens=True)[0]

            whisper_transcription[idx] = str(hyp)
            whisper_wer[idx] = _wer(gt, hyp)
            decoded += 1

            if decoded % 100 == 0 or decoded == n:
                print(f"Decoded {decoded}/{n} ({100.0 * decoded / n:.1f}%)")

        global_offset += rg_n

    out = Path(args.output_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out),
        whisper_wer=whisper_wer,
        whisper_transcription=np.asarray(whisper_transcription, dtype=object),
    )
    test_mean = float(whisper_wer[:n].mean())
    print(f"Wrote overrides: {out}")
    print(f"All-clips whisper_wer mean: {test_mean * 100:.2f}%")


if __name__ == "__main__":
    main()
