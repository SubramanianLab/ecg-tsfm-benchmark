"""Forecasting workflows."""
from __future__ import annotations

import argparse
from typing import Any, Dict, List

import numpy as np

from ecg_dataloader import BEAT_ANNOTATION_SYMBOLS, load_record_window, resolve_record_path
from ecg_rr import _build_exact_rr_window, _load_full_annotation_samples
from models import run_chronos, run_moirai2, run_timesfm

def _forecast_with_model(
    inputs: List[np.ndarray],
    horizon: int,
    context_length: int,
    model_name: str,
    moirai2_device: str = "cpu",
    moirai2_batch_size: int = 32,
) -> Dict[str, Any]:
    if model_name == "timesfm":
        return run_timesfm(inputs, horizon, context_length)
    if model_name == "chronos":
        return run_chronos(inputs, horizon, context_length)
    if model_name == "moirai2":
        return run_moirai2(inputs, horizon, context_length, device=moirai2_device, batch_size=moirai2_batch_size)
    raise ValueError(f"Unsupported model: {model_name}")
def forecast_records(
    records: List[Dict[str, Any]],
    horizon: int,
    context_length: int,
    rr_context_beats: int,
    rr_horizon_beats: int,
    model_name: str,
    moirai2_device: str = "cpu",
    moirai2_batch_size: int = 32,
) -> List[Dict[str, Any]]:
    waveform_inputs: List[np.ndarray] = []
    rr_inputs: List[np.ndarray] = []
    for record in records:
        signals = record["signals"]
        if "rr_context" in record:
            rr_context = np.asarray(record["rr_context"], dtype=np.float32)
        else:
            full_rpeaks_abs = np.asarray(record.get("full_annotation_samples_abs", []), dtype=np.int64)
            forecast_boundary_abs = int(record["sample_offset"]) + context_length
            full_rr_intervals = (
                np.diff(full_rpeaks_abs).astype(np.float32)
                if len(full_rpeaks_abs) > 1
                else np.array([], dtype=np.float32)
            )
            boundary_peak_idx = int(np.searchsorted(full_rpeaks_abs, forecast_boundary_abs, side="left"))
            boundary_rr_idx = max(0, boundary_peak_idx - 1)
            rr_context_start = max(0, boundary_rr_idx - rr_context_beats)
            rr_context = full_rr_intervals[rr_context_start:boundary_rr_idx]
        for lead_index in range(len(record["lead_names"])):
            waveform_inputs.append(signals[lead_index, :context_length])
            rr_inputs.append(rr_context)
    waveform_result = _forecast_with_model(
        waveform_inputs,
        horizon,
        context_length,
        model_name,
        moirai2_device,
        moirai2_batch_size,
    )
    rr_result = _forecast_with_model(
        rr_inputs,
        rr_horizon_beats,
        rr_context_beats,
        model_name,
        moirai2_device,
        moirai2_batch_size,
    )
    grouped_results: List[Dict[str, Any]] = []
    cursor = 0
    for record in records:
        num_record_leads = len(record["lead_names"])
        point_all = []
        rr_point_all = []
        for lead_index in range(num_record_leads):
            point_all.append(waveform_result["point"][cursor + lead_index])
            rr_point_all.append(rr_result["point"][cursor + lead_index])
        grouped_results.append(
            {
                "record_id": record["record_id"],
                "lead_names": record["lead_names"],
                "name": waveform_result["name"],
                "color": waveform_result["color"],
                "point": np.stack(point_all, axis=0),
                "rr_point": np.stack(rr_point_all, axis=0),
            }
        )
        cursor += num_record_leads
    return grouped_results

def prepare_forecast_window(args: argparse.Namespace) -> Dict[str, Any]:
    record_stem = resolve_record_path(args.data_root, args.record)
    total_window = args.waveform_context + args.waveform_horizon
    start_sample = max(0, args.start_sample)
    window = load_record_window(
        record_stem,
        leads=args.requested_leads,
        sampfrom=start_sample,
        sampto=start_sample + total_window,
        normalize=args.normalize,
        load_annotations=True,
        annotation_symbols_filter=BEAT_ANNOTATION_SYMBOLS,
    )
    if window["n_samples"] < total_window:
        raise ValueError(
            f"Requested {total_window} samples from offset {start_sample}, but only "
            f"{window['n_samples']} are available."
        )
    if window["missing_leads"]:
        available = ", ".join(window["available_leads"])
        missing = ", ".join(window["missing_leads"])
        raise ValueError(f"Missing lead(s): {missing}. Available leads for this record: {available}")
    window["context_length"] = args.waveform_context
    window["horizon"] = args.waveform_horizon
    window["full_annotation_samples_abs"] = _load_full_annotation_samples(record_stem)
    window["record_length"] = int(
        max(
            window["n_samples"] + window["sample_offset"],
            np.max(window["full_annotation_samples_abs"]) + 1
            if len(window["full_annotation_samples_abs"])
            else window["n_samples"] + window["sample_offset"],
        )
    )
    if args.paper_rr_figure:
        window.update(
            _build_exact_rr_window(
                record_stem,
                args.rr_context,
                args.rr_horizon,
                start_sample=args.start_sample,
            )
        )
    return window
