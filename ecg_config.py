"""Shared configuration and path helpers for ECG forecasting."""
from __future__ import annotations

from pathlib import Path
from typing import List

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

DATASET_ROOTS = {
    "arrhythmia": "mit-bih-arrhythmia-database-1.0.0",
    "nsrdb": "mit-bih-normal-sinus-rhythm-database-1.0.0",
}
MIN_RR_CONTEXT_BEATS = 512
DEFAULT_RR_CONTEXT_BEATS = 2000

def _resolve_default_data_root(dataset: str) -> Path:
    direct_path = ROOT_DIR / DATASET_ROOTS[dataset]
    if direct_path.exists():
        return direct_path
    archive_path = ROOT_DIR / "Archive" / DATASET_ROOTS[dataset]
    if archive_path.exists():
        return archive_path
    return direct_path


def _resolve_output_path(path_str: str) -> Path:
    path_str = _normalize_workspace_path(path_str)
    path = Path(path_str)
    if path.is_absolute():
        return path
    return SCRIPT_DIR / path


def _normalize_workspace_path(path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return path_str
    parts = path.parts
    if len(parts) >= 2 and parts[0].lower() == "archive" and parts[1] == "figures":
        return str(Path(*parts[1:]))
    return path_str


def _resolve_input_path(path_str: str) -> Path:
    path = Path(_normalize_workspace_path(path_str))
    if path.is_absolute():
        return path
    return SCRIPT_DIR / path


def _require_input_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def _default_output_name(dataset: str, records: List[str]) -> str:
    sample = Path(records[0]).stem if len(records) == 1 else "multi"
    output_dir = SCRIPT_DIR / "figures" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir / f"mit_bih_{dataset}_{sample}.png")
def _parse_int_list(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]
