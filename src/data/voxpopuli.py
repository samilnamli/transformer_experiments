"""VoxPopuli DataModule.

Uses the bundled processed parquet at
``configs/data/processed/facebook_voxpopuli/`` when present. Falls back
to ``data/processed/facebook_voxpopuli/`` (Google Drive download target)
when the bundled copy is absent.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.data.base import ASRDataModule
from src.utils.git import REPO_ROOT

VOXPOPULI_PARQUET_NAME = "combined_features_with_transcripts.parquet"
VOXPOPULI_BUNDLED_PARQUET = (
    REPO_ROOT / "configs/data/processed/facebook_voxpopuli" / VOXPOPULI_PARQUET_NAME
)
VOXPOPULI_LEGACY_PARQUET = (
    REPO_ROOT / "data/processed/facebook_voxpopuli" / VOXPOPULI_PARQUET_NAME
)
VOXPOPULI_DEFAULT_PARQUET = str(VOXPOPULI_BUNDLED_PARQUET)
VOXPOPULI_DRIVE_FILE_ID = "1yf-G-DWhhZLlqeGXZ77GbuhTBmhqyyXA"


def resolve_voxpopuli_parquet(parquet_path: str | None = None) -> Path:
    """Return the first existing VoxPopuli parquet, or the download target.

    Search order:
        1. Explicit ``parquet_path`` (repo-root-relative or absolute)
        2. Bundled copy under ``configs/data/processed/facebook_voxpopuli/``
        3. Legacy download location under ``data/processed/facebook_voxpopuli/``
        4. Explicit path even when missing (so callers can create/download it)
    """
    candidates: list[Path] = []
    if parquet_path:
        configured = Path(parquet_path)
        if not configured.is_absolute():
            configured = REPO_ROOT / configured
        candidates.append(configured)
    candidates.extend([VOXPOPULI_BUNDLED_PARQUET, VOXPOPULI_LEGACY_PARQUET])

    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    return candidates[0].resolve()


class VoxPopuliDataModule(ASRDataModule):
    """ASRDataModule specialised for VoxPopuli (en_accented test split)."""

    def __init__(
        self,
        parquet_path: str = VOXPOPULI_DEFAULT_PARQUET,
        auto_download: bool = False,
        drive_file_id: str = VOXPOPULI_DRIVE_FILE_ID,
        **kwargs,
    ):
        resolved = resolve_voxpopuli_parquet(parquet_path)
        super().__init__(parquet_path=str(resolved), **kwargs)
        self._configured_parquet_path = parquet_path
        self.auto_download = auto_download
        self.drive_file_id = drive_file_id

    def prepare_data(self) -> None:
        path = resolve_voxpopuli_parquet(self._configured_parquet_path)
        if path.exists():
            self.parquet_path = str(path)
            logger.info("VoxPopuli parquet ready at {}", path)
            return

        if self.auto_download:
            from src.data.downloads import download_drive_file

            target = VOXPOPULI_LEGACY_PARQUET
            logger.info("VoxPopuli parquet missing — downloading from Google Drive ({})", target)
            target.parent.mkdir(parents=True, exist_ok=True)
            download_drive_file(self.drive_file_id, str(target))
            if target.exists():
                self.parquet_path = str(target)
                return

        raise FileNotFoundError(
            f"VoxPopuli parquet not found. Checked bundled path "
            f"({VOXPOPULI_BUNDLED_PARQUET}), legacy path "
            f"({VOXPOPULI_LEGACY_PARQUET}), and configured path "
            f"({self._configured_parquet_path}). Place the processed parquet "
            f"under configs/data/processed/facebook_voxpopuli/ or set "
            f"auto_download=true to fetch from Google Drive."
        )
