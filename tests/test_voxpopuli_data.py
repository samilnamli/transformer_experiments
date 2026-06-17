"""Tests for the bundled VoxPopuli processed parquet."""

from __future__ import annotations

import pyarrow.parquet as pq
import pytest

from src.data.voxpopuli import (
    VOXPOPULI_BUNDLED_PARQUET,
    VoxPopuliDataModule,
    resolve_voxpopuli_parquet,
)


@pytest.fixture(scope="module")
def bundled_parquet() -> str:
    path = resolve_voxpopuli_parquet()
    if not path.exists():
        pytest.skip(f"Bundled VoxPopuli parquet not found at {path}")
    return str(path)


def test_resolve_prefers_bundled_copy(bundled_parquet: str) -> None:
    resolved = resolve_voxpopuli_parquet()
    assert resolved.exists()
    assert resolved == VOXPOPULI_BUNDLED_PARQUET.resolve()


def test_prepare_data_uses_bundled_parquet(bundled_parquet: str) -> None:
    dm = VoxPopuliDataModule(auto_download=False, batch_size=8, num_workers=0)
    dm.prepare_data()
    assert dm.parquet_path == bundled_parquet


def test_bundled_parquet_has_legacy_schema(bundled_parquet: str) -> None:
    schema = pq.read_schema(bundled_parquet)
    required = {
        "ground_truth",
        "hubert_features",
        "whisper_features",
        "w2v2_features",
        "hubert_wer",
        "whisper_wer",
        "w2v2_wer",
    }
    assert required <= set(schema.names)


def test_single_whisper_wer_on_test_split(bundled_parquet: str) -> None:
    dm = VoxPopuliDataModule(
        parquet_path=bundled_parquet,
        auto_download=False,
        seed=42,
        whisper_decode_overrides_path=None,
    )
    dm.prepare_data()
    dm.setup("test")

    wer = dm._dataset.wer_matrix[dm.test_dataset.indices]
    whisper_mean = float(wer[:, 1].mean())
    assert wer.shape[0] == len(dm.test_dataset)
    assert 0.0 <= whisper_mean <= 5.0
    # Legacy bundled parquet (pre-fix decode) matches manuscript Whisper WER.
    assert whisper_mean == pytest.approx(1.214, rel=0.05)
