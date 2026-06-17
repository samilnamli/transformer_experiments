"""Quick VoxPopuli parquet sanity check (run via ``uv run python scripts/check_voxpopuli.py``)."""

from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq

from src.data.voxpopuli import VoxPopuliDataModule, resolve_voxpopuli_parquet


def main() -> None:
    path = resolve_voxpopuli_parquet()
    print(f"parquet: {path}")
    meta = pq.read_metadata(path)
    print(f"rows: {meta.num_rows}")

    wer_cols = ["whisper_wer", "hubert_wer", "w2v2_wer"]
    table = pq.read_table(path, columns=wer_cols)
    for col in wer_cols:
        wer = table[col].to_numpy()
        print(
            f"{col}: mean={wer.mean() * 100:.2f}% "
            f"median={np.median(wer) * 100:.2f}% "
            f"max={wer.max() * 100:.2f}% "
            f">100%={(wer > 1).sum()}"
        )

    dm = VoxPopuliDataModule(auto_download=False, seed=42)
    dm.prepare_data()
    dm.setup("test")
    test_idx = dm.test_dataset.indices
    wer = dm._dataset.wer_matrix[test_idx]
    for idx, name in enumerate(["hubert", "whisper", "wav2vec2"]):
        mean = float(wer[:, idx].mean())
        print(f"single_{name} test wer_mean={mean * 100:.2f}% n={len(test_idx)}")


if __name__ == "__main__":
    main()
