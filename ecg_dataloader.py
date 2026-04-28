"""
MIT-BIH ECG data loading utilities.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
COMMON_MITBIH_LEADS = ["MLII", "V1", "V2", "V4", "V5"]
BEAT_ANNOTATION_SYMBOLS = {
    "N",
    "L",
    "R",
    "B",
    "A",
    "a",
    "J",
    "S",
    "V",
    "r",
    "F",
    "e",
    "j",
    "n",
    "E",
    "/",
    "f",
    "Q",
    "?",
}
def _require_wfdb():
    try:
        import wfdb
    except ImportError as exc:
        raise ImportError(
            "The `wfdb` package is required to read MIT-BIH records. Install it with: pip install wfdb"
        ) from exc
    return wfdb
def _parse_patient_metadata(comments: Sequence[str]) -> Dict[str, Optional[Union[int, str]]]:
    age: Optional[int] = None
    sex: Optional[str] = None
    for comment in comments:
        tokens = comment.replace("#", " ").split()
        if len(tokens) >= 2:
            if age is None:
                try:
                    age = int(tokens[0])
                except ValueError:
                    pass
            if sex is None and tokens[1] in {"M", "F"}:
                sex = tokens[1]
        if age is not None or sex is not None:
            break
    return {"patient_age": age, "patient_sex": sex}
def _ordered_leads(leads: Sequence[str]) -> List[str]:
    unique: List[str] = []
    seen = set()
    for lead in leads:
        name = lead.strip()
        if name and name not in seen:
            unique.append(name)
            seen.add(name)
    priority = {lead: idx for idx, lead in enumerate(COMMON_MITBIH_LEADS)}
    return sorted(unique, key=lambda lead: (priority.get(lead, 999), lead))


def _annotation_mask(symbols: Sequence[str], symbols_filter: Optional[Sequence[str]]) -> np.ndarray:
    if symbols_filter is None:
        return np.ones(len(symbols), dtype=bool)
    allowed = set(symbols_filter)
    return np.asarray([symbol in allowed for symbol in symbols], dtype=bool)


def list_mitbih_records(data_root: Union[str, Path]) -> List[str]:
    root = Path(data_root)
    records_file = root / "RECORDS"
    if records_file.exists():
        return [line.strip() for line in records_file.read_text().splitlines() if line.strip()]
    return sorted(path.stem for path in root.glob("*.hea"))
def resolve_record_path(data_root: Union[str, Path], record_id_or_path: str) -> Path:
    root = Path(data_root)
    raw = Path(record_id_or_path)
    if raw.suffix in {".hea", ".dat", ".atr"}:
        raw = raw.with_suffix("")
    if raw.is_absolute():
        return raw
    candidate = root / raw
    if candidate.with_suffix(".hea").exists():
        return candidate
    candidate = root / raw.name
    if candidate.with_suffix(".hea").exists():
        return candidate
    raise FileNotFoundError(
        f"Could not resolve MIT-BIH record '{record_id_or_path}' under {root}"
    )
def inspect_record(record_path: Union[str, Path]) -> Dict[str, Any]:
    wfdb = _require_wfdb()
    stem = Path(record_path)
    header = wfdb.rdheader(str(stem))
    lead_names = [name.strip() for name in header.sig_name]
    patient_meta = _parse_patient_metadata(header.comments or [])
    return {
        "record_id": stem.name,
        "path": str(stem),
        "sig_len": int(header.sig_len),
        "sampling_rate": int(round(float(header.fs))),
        "available_leads": lead_names,
        "comments": list(header.comments or []),
        **patient_meta,
    }
def load_record_window(
    record_path: Union[str, Path],
    leads: Union[str, Sequence[str]] = "all",
    sampfrom: int = 0,
    sampto: Optional[int] = None,
    normalize: bool = False,
    load_annotations: bool = False,
    annotation_extension: str = "atr",
    annotation_symbols_filter: Optional[Sequence[str]] = None,
    dtype: str = "float32",
) -> Dict[str, Any]:
    wfdb = _require_wfdb()
    stem = Path(record_path)
    meta = inspect_record(stem)
    requested_leads = (
        _ordered_leads(meta["available_leads"])
        if leads == "all"
        else [lead.strip() for lead in leads]
    )
    available_map = {lead: idx for idx, lead in enumerate(meta["available_leads"])}
    total_length = meta["sig_len"]
    start = max(0, int(sampfrom))
    stop = total_length if sampto is None else min(int(sampto), total_length)
    if stop <= start:
        raise ValueError(f"Invalid sample range [{start}, {stop}) for record {stem.name}")
    channel_indices = [available_map[lead] for lead in requested_leads if lead in available_map]
    present_leads = [lead for lead in requested_leads if lead in available_map]
    missing_leads = [lead for lead in requested_leads if lead not in available_map]
    signals = np.zeros((len(requested_leads), stop - start), dtype=getattr(np, dtype))
    if channel_indices:
        record = wfdb.rdrecord(
            str(stem),
            sampfrom=start,
            sampto=stop,
            channels=channel_indices,
        )
        loaded = record.p_signal.astype(getattr(np, dtype), copy=False)
        lead_to_pos = {lead: pos for pos, lead in enumerate(requested_leads)}
        for i, lead in enumerate(present_leads):
            signals[lead_to_pos[lead], : loaded.shape[0]] = loaded[:, i]
    if normalize:
        for row in range(signals.shape[0]):
            segment = signals[row]
            std = float(segment.std())
            if std > 1e-8:
                signals[row] = (segment - float(segment.mean())) / std
    result: Dict[str, Any] = {
        "signals": signals,
        "lead_names": requested_leads,
        "available_leads": meta["available_leads"],
        "missing_leads": missing_leads,
        "record_id": meta["record_id"],
        "record_path": str(stem),
        "sampling_rate": meta["sampling_rate"],
        "n_samples": stop - start,
        "sample_offset": start,
        "comments": meta["comments"],
        "patient_age": meta["patient_age"],
        "patient_sex": meta["patient_sex"],
    }
    if load_annotations:
        ann = wfdb.rdann(str(stem), annotation_extension, sampfrom=start, sampto=stop)
        keep = _annotation_mask(ann.symbol, annotation_symbols_filter)
        result["annotation_samples"] = ann.sample[keep].astype(np.int64) - start
        result["annotation_symbols"] = [symbol for symbol, include in zip(ann.symbol, keep) if include]
        result["annotation_aux_note"] = [note for note, include in zip(ann.aux_note, keep) if include]
    return result
