"""R-peak and RR interval helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from ecg_dataloader import BEAT_ANNOTATION_SYMBOLS

def _load_full_annotation_samples(record_path: Path, annotation_extension: str = "atr") -> np.ndarray:
    try:
        import wfdb
    except ImportError as exc:
        raise ImportError(
            "The `wfdb` package is required to read MIT-BIH annotations. Install it with: pip install wfdb"
        ) from exc
    ann = wfdb.rdann(str(record_path), annotation_extension)
    keep = np.asarray([symbol in BEAT_ANNOTATION_SYMBOLS for symbol in ann.symbol], dtype=bool)
    return ann.sample[keep].astype(np.int64)

def _build_exact_rr_window(
    record_path: Path,
    rr_context_beats: int,
    rr_horizon_beats: int,
    *,
    start_sample: int = 0,
) -> Dict[str, Any]:
    full_rpeaks_abs = _load_full_annotation_samples(record_path)
    full_rr_intervals = (
        np.diff(full_rpeaks_abs).astype(np.float32)
        if len(full_rpeaks_abs) > 1
        else np.array([], dtype=np.float32)
    )
    total_rr = int(len(full_rr_intervals))
    required = int(rr_context_beats + rr_horizon_beats)
    if total_rr < required:
        raise ValueError(
            f"Record {record_path.stem} has only {total_rr} RR intervals, "
            f"but {required} are required for rr-context={rr_context_beats} "
            f"and rr-horizon={rr_horizon_beats}."
        )

    if start_sample > 0:
        boundary_rr_idx = int(np.searchsorted(full_rpeaks_abs[1:], start_sample, side="left"))
        boundary_rr_idx = max(int(rr_context_beats), boundary_rr_idx)
    else:
        boundary_rr_idx = int(rr_context_beats)

    max_boundary_rr_idx = total_rr - int(rr_horizon_beats)
    if boundary_rr_idx > max_boundary_rr_idx:
        boundary_rr_idx = max_boundary_rr_idx

    context_start = boundary_rr_idx - int(rr_context_beats)
    future_stop = boundary_rr_idx + int(rr_horizon_beats)
    if context_start < 0 or future_stop > total_rr:
        raise ValueError(
            f"Could not construct exact RR window for record {record_path.stem} "
            f"with rr-context={rr_context_beats}, rr-horizon={rr_horizon_beats}, "
            f"start_sample={start_sample}."
        )

    return {
        "full_annotation_samples_abs": full_rpeaks_abs,
        "full_rr_intervals": full_rr_intervals,
        "rr_boundary_index": int(boundary_rr_idx),
        "rr_context_start": int(context_start),
        "rr_context": full_rr_intervals[context_start:boundary_rr_idx].copy(),
        "rr_future": full_rr_intervals[boundary_rr_idx:future_stop].copy(),
    }
