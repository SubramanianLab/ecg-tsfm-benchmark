from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from ecg_config import _resolve_default_data_root
from ecg_dataloader import BEAT_ANNOTATION_SYMBOLS, inspect_record, load_record_window, resolve_record_path
from ecg_rr import _load_full_annotation_samples
from ecg_workflows import forecast_records


METRIC_FIELDNAMES = [
    "sweep_type",
    "model",
    "context_length",
    "horizon",
    "rr_context_beats",
    "rr_horizon_beats",
    "rmse",
    "mae",
    "window_rmse_mean",
    "window_mae_mean",
    "num_windows",
    "num_series",
    "rr_rmse",
    "rr_mae",
    "rr_window_rmse_mean",
    "rr_window_mae_mean",
    "rr_ks_mean",
    "rr_ks_cutoff_mean",
    "rr_ks_pass_rate",
    "waveform_rr_rmse",
    "waveform_rr_mae",
    "waveform_rr_window_rmse_mean",
    "waveform_rr_window_mae_mean",
    "waveform_rr_ks_mean",
    "waveform_rr_ks_cutoff_mean",
    "waveform_rr_ks_pass_rate",
]

STEP_FIELDNAMES = [
    "sweep_type",
    "metric_type",
    "model",
    "context_length",
    "horizon",
    "rr_context_beats",
    "rr_horizon_beats",
    "step_index",
    "rmse",
    "mae",
    "count",
]

MODEL_NAMES = {
    "timesfm": "TimesFM",
    "chronos": "Chronos-2",
    "moirai2": "Moirai 2.0",
}

NSRDB_SUBJECT_IDS = (
    "16265",
    "16272",
    "16273",
    "16420",
    "16483",
    "16539",
    "16773",
    "16786",
    "16795",
    "17052",
    "17453",
    "18177",
    "18184",
    "19088",
    "19090",
    "19093",
    "19140",
    "19830",
)

SUBJECT_METRIC_FIELDNAMES = ["subject_id", *METRIC_FIELDNAMES]
SUBJECT_STEP_FIELDNAMES = ["subject_id", *STEP_FIELDNAMES]

METRIC_KEY_FIELDS = [
    "model",
    "context_length",
    "horizon",
    "rr_context_beats",
    "rr_horizon_beats",
]

STEP_KEY_FIELDS = [
    "model",
    "context_length",
    "horizon",
    "rr_context_beats",
    "rr_horizon_beats",
    "step_index",
]

SUM_FIELDS = {"num_windows", "num_series", "count"}


def is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "nan", "none", "null", "na", "n/a"}:
        return True
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def csv_value(value: Any) -> Any:
    return "NA" if is_missing_value(value) else value


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: csv_value(row.get(key)) for key in fieldnames} for row in rows])


def merge_model_rows(path: Path, new_rows: Sequence[Dict[str, Any]], model_names: Sequence[str], fieldnames: Sequence[str]) -> None:
    old_rows = read_rows(path)
    excluded = set(model_names)
    kept = [row for row in old_rows if row.get("model") not in excluded]
    write_rows(path, [*kept, *new_rows], fieldnames)


def resolve_subject_ids(record_ids: Sequence[str]) -> List[str]:
    if len(record_ids) == 1 and record_ids[0].strip().lower() == "all":
        return list(NSRDB_SUBJECT_IDS)
    return [record_id.strip() for record_id in record_ids if record_id.strip()]


def _cache_path(results_dir: Path, subject_id: str, suffix: str) -> Path:
    safe_subject = subject_id.replace("/", "_")
    return results_dir / f"subject_{safe_subject}_{suffix}.csv"


def _as_int(value: Any) -> int:
    return int(float(value))


def _as_float(value: Any) -> float:
    if is_missing_value(value):
        return math.nan
    return float(value)


def _row_key(row: Dict[str, Any], fields: Sequence[str]) -> tuple[Any, ...]:
    values = []
    for field in fields:
        if field == "model":
            values.append(str(row[field]))
        else:
            values.append(_as_int(row[field]))
    return tuple(values)


def _metric_cache_key(model_name: str, context_length: int, horizon: int, rr_context: int, rr_horizon: int) -> tuple[Any, ...]:
    return (
        model_name,
        int(context_length),
        int(horizon),
        int(rr_context),
        int(rr_horizon),
    )


def _metric_row_has_ks(row: Dict[str, Any]) -> bool:
    return not (
        is_missing_value(row.get("rr_ks_mean"))
        or is_missing_value(row.get("waveform_rr_ks_mean"))
    )


def _sort_metric_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    model_order = {name: index for index, name in enumerate(MODEL_NAMES.values())}
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("subject_id", "")),
            _as_int(row["context_length"]),
            _as_int(row["horizon"]),
            model_order.get(str(row["model"]), 999),
        ),
    )


def _sort_step_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    model_order = {name: index for index, name in enumerate(MODEL_NAMES.values())}
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("subject_id", "")),
            _as_int(row["context_length"]),
            _as_int(row["horizon"]),
            model_order.get(str(row["model"]), 999),
            _as_int(row["step_index"]),
        ),
    )


def _average_subject_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    key_fields: Sequence[str],
    fieldnames: Sequence[str],
) -> List[Dict[str, Any]]:
    groups: Dict[tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_row_key(row, key_fields), []).append(row)

    averaged_rows: List[Dict[str, Any]] = []
    for key, group_rows in groups.items():
        output: Dict[str, Any] = {"sweep_type": "waveform_context"}
        for field, value in zip(key_fields, key):
            output[field] = value
        if "metric_type" in fieldnames:
            output["metric_type"] = "rr"

        for field in fieldnames:
            if field in output or field in key_fields:
                continue
            if field in {"sweep_type", "metric_type"}:
                continue
            if field in SUM_FIELDS:
                output[field] = int(sum(_as_int(row[field]) for row in group_rows if not is_missing_value(row.get(field))))
                continue
            values = [_as_float(row.get(field)) for row in group_rows]
            finite_values = [value for value in values if not math.isnan(value)]
            output[field] = float(np.mean(finite_values)) if finite_values else math.nan

        averaged_rows.append(output)

    model_order = {name: index for index, name in enumerate(MODEL_NAMES.values())}
    return sorted(
        averaged_rows,
        key=lambda row: (
            _as_int(row["context_length"]),
            _as_int(row["horizon"]),
            model_order.get(str(row["model"]), 999),
            _as_int(row.get("step_index", 0)) if "step_index" in row else 0,
        ),
    )


def build_evaluation_records(
    data_root: Path,
    record_ids: Sequence[str],
    context_length: int,
    horizon: int,
    max_windows: int,
) -> List[Dict[str, Any]]:
    total_window = int(context_length) + int(horizon)
    records: List[Dict[str, Any]] = []
    for record_id in record_ids:
        record_stem = resolve_record_path(data_root, record_id)
        sig_len = int(inspect_record(record_stem)["sig_len"])
        last_start = sig_len - total_window
        if last_start < 0:
            continue
        offsets = list(range(0, last_start + 1, horizon))
        if offsets and offsets[-1] != last_start:
            offsets.append(last_start)
        for start_sample in offsets:
            window = load_record_window(
                record_stem,
                leads="all",
                sampfrom=start_sample,
                sampto=start_sample + total_window,
                normalize=False,
                load_annotations=True,
                annotation_symbols_filter=BEAT_ANNOTATION_SYMBOLS,
            )
            if window["n_samples"] != total_window or window["missing_leads"]:
                continue
            window["context_length"] = int(context_length)
            window["horizon"] = int(horizon)
            window["full_annotation_samples_abs"] = _load_full_annotation_samples(record_stem)
            records.append(window)
            if len(records) >= max_windows:
                return records
    return records


def future_rr_from_annotations(record: Dict[str, Any], context_length: int, rr_horizon: int) -> np.ndarray:
    full_rpeaks = np.asarray(record["full_annotation_samples_abs"], dtype=np.int64)
    full_rr = np.diff(full_rpeaks).astype(np.float32) if len(full_rpeaks) > 1 else np.array([], dtype=np.float32)
    boundary = int(record["sample_offset"]) + int(context_length)
    boundary_peak_idx = int(np.searchsorted(full_rpeaks, boundary, side="left"))
    boundary_rr_idx = max(0, boundary_peak_idx - 1)
    return full_rr[boundary_rr_idx : boundary_rr_idx + int(rr_horizon)]


def future_rr_for_waveform_window(record: Dict[str, Any], context_length: int, horizon: int) -> np.ndarray:
    full_rpeaks = np.asarray(record["full_annotation_samples_abs"], dtype=np.int64)
    if len(full_rpeaks) < 2:
        return np.array([], dtype=np.float32)
    boundary = int(record["sample_offset"]) + int(context_length)
    stop = boundary + int(horizon)
    second_peak = full_rpeaks[1:]
    mask = (second_peak >= boundary) & (second_peak < stop)
    return np.diff(full_rpeaks).astype(np.float32)[mask]


def _series_polarity(context_signal: np.ndarray) -> float:
    signal = np.asarray(context_signal, dtype=np.float32)
    if signal.size == 0:
        return 1.0
    center = float(np.median(signal))
    upper = float(np.percentile(signal, 99) - center)
    lower = float(center - np.percentile(signal, 1))
    return 1.0 if upper >= lower else -1.0


def extract_rr_from_waveform(
    context_signal: np.ndarray,
    predicted_signal: np.ndarray,
    *,
    sampling_rate: float,
) -> np.ndarray:
    try:
        from scipy.signal import find_peaks
    except ImportError as exc:
        raise ImportError("scipy is required to extract RR intervals from waveform forecasts.") from exc

    context = np.asarray(context_signal, dtype=np.float32).reshape(-1)
    predicted = np.asarray(predicted_signal, dtype=np.float32).reshape(-1)
    if context.size == 0 or predicted.size == 0:
        return np.array([], dtype=np.float32)

    combined = np.concatenate([context, predicted])
    oriented = combined * _series_polarity(context)
    tail = oriented[max(0, context.size - int(round(8 * sampling_rate))) : context.size]
    scale_source = tail if tail.size else oriented
    amplitude = float(np.percentile(scale_source, 99) - np.percentile(scale_source, 5))
    distance = max(1, int(round(0.3 * sampling_rate)))
    prominence = max(1e-6, 0.12 * amplitude, 0.02 * float(np.std(scale_source)))
    peaks, _ = find_peaks(oriented, distance=distance, prominence=prominence)
    if len(peaks) < 2:
        return np.array([], dtype=np.float32)
    rr = np.diff(peaks).astype(np.float32)
    second_peak = peaks[1:]
    return rr[(second_peak >= context.size) & (second_peak < combined.size)]


def waveform_pairs(
    records: Sequence[Dict[str, Any]],
    results: Sequence[Dict[str, Any]],
    context_length: int,
    horizon: int,
) -> List[tuple[np.ndarray, np.ndarray]]:
    pairs: List[tuple[np.ndarray, np.ndarray]] = []
    for record, result in zip(records, results):
        truth = future_rr_for_waveform_window(record, context_length, horizon)
        sampling_rate = float(record["sampling_rate"])
        for lead_index in range(len(record["lead_names"])):
            pred = extract_rr_from_waveform(
                record["signals"][lead_index, :context_length],
                result["point"][lead_index],
                sampling_rate=sampling_rate,
            )
            limit = min(len(truth), len(pred))
            if limit:
                pairs.append((truth[:limit], pred[:limit]))
    return pairs


def direct_rr_pairs(
    records: Sequence[Dict[str, Any]],
    results: Sequence[Dict[str, Any]],
    context_length: int,
    rr_horizon: int,
) -> List[tuple[np.ndarray, np.ndarray]]:
    pairs: List[tuple[np.ndarray, np.ndarray]] = []
    for record, result in zip(records, results):
        truth = future_rr_from_annotations(record, context_length, rr_horizon)
        for lead_index in range(len(record["lead_names"])):
            pred = np.asarray(result["rr_point"][lead_index], dtype=np.float32)[: len(truth)]
            limit = min(len(truth), len(pred))
            if limit:
                pairs.append((truth[:limit], pred[:limit]))
    return pairs



def direct_rr_pit_pairs(
    records: Sequence[Dict[str, Any]],
    results: Sequence[Dict[str, Any]],
    context_length: int,
    rr_horizon: int,
) -> List[tuple[np.ndarray, np.ndarray]]:
    """Return (truth, z) pairs for direct RR PIT/time-rescaling KS.

    z must be F(true_rr_i | history). forecast_records must provide one of:
    rr_pit/rr_cdf/rr_z, rr_samples, or rr_quantiles plus quantile levels. If
    only rr_point exists, this returns no pairs and KS remains NA.
    """
    pairs: List[tuple[np.ndarray, np.ndarray]] = []
    for record, result in zip(records, results):
        truth = future_rr_from_annotations(record, context_length, rr_horizon)
        for lead_index in range(len(record["lead_names"])):
            pit = _pit_values_from_result(result, lead_index, truth, prefix="rr")
            limit = min(len(truth), len(pit))
            if limit:
                pairs.append((truth[:limit], pit[:limit]))
    return pairs


def waveform_rr_pit_pairs(
    records: Sequence[Dict[str, Any]],
    results: Sequence[Dict[str, Any]],
    context_length: int,
    horizon: int,
) -> List[tuple[np.ndarray, np.ndarray]]:
    """Return (truth, z) pairs for waveform-derived RR PIT/time-rescaling KS.

    This requires forecast_records to provide waveform_rr_pit/waveform_rr_cdf/
    waveform_rr_z, waveform_rr_samples, or waveform_rr_quantiles. Extracting a
    single RR sequence from a point waveform forecast is not enough to compute
    a time-rescaling KS statistic.
    """
    pairs: List[tuple[np.ndarray, np.ndarray]] = []
    for record, result in zip(records, results):
        truth = future_rr_for_waveform_window(record, context_length, horizon)
        sampling_rate = float(record["sampling_rate"])
        for lead_index in range(len(record["lead_names"])):
            pit = _pit_values_from_result(result, lead_index, truth, prefix="waveform_rr")
            if pit.size == 0:
                pit = _waveform_rr_pit_from_waveform_quantiles(
                    record,
                    result,
                    lead_index,
                    truth,
                    context_length=context_length,
                    sampling_rate=sampling_rate,
                )
            limit = min(len(truth), len(pit))
            if limit:
                pairs.append((truth[:limit], pit[:limit]))
    return pairs

def waveform_forecast_pairs(
    records: Sequence[Dict[str, Any]],
    results: Sequence[Dict[str, Any]],
    context_length: int,
    horizon: int,
) -> List[tuple[np.ndarray, np.ndarray]]:
    pairs: List[tuple[np.ndarray, np.ndarray]] = []
    for record, result in zip(records, results):
        future = np.asarray(record["signals"][:, context_length : context_length + horizon], dtype=np.float32)
        for lead_index in range(len(record["lead_names"])):
            pred = np.asarray(result["point"][lead_index], dtype=np.float32)[: future.shape[1]]
            limit = min(future.shape[1], len(pred))
            if limit:
                pairs.append((future[lead_index, :limit], pred[:limit]))
    return pairs


def _empty_metric_dict() -> Dict[str, Any]:
    return {
        "rmse": math.nan,
        "mae": math.nan,
        "window_rmse_mean": math.nan,
        "window_mae_mean": math.nan,
        "ks_mean": math.nan,
        "ks_cutoff_mean": math.nan,
        "ks_pass_rate": math.nan,
    }


def pit_ks_from_cdf_values(cdf_values: np.ndarray) -> Dict[str, float]:
    """One-sample KS statistic against Uniform(0, 1) for PIT/time-rescaled values.

    Each value should be z_i = F(tau_i | H_{i-1}), where tau_i is the observed
    RR interval and F is the model's conditional CDF for that interval.
    """
    z = np.asarray(cdf_values, dtype=np.float64).reshape(-1)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return {"ks": math.nan, "ks_cutoff": math.nan, "ks_pass": math.nan}

    z = np.clip(z, 1e-7, 1.0 - 1e-7)
    z_sorted = np.sort(z)
    n = int(z_sorted.size)
    reference = (np.arange(1, n + 1, dtype=np.float64) - 0.5) / n
    ks = float(np.max(np.abs(reference - z_sorted)))
    cutoff = float(1.36 / np.sqrt(n))
    return {"ks": ks, "ks_cutoff": cutoff, "ks_pass": float(ks <= cutoff)}


def empirical_cdf_values_from_samples(samples: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Estimate F(observed_i | H_{i-1}) from forecast samples.

    Accepted sample shapes are (num_samples, horizon) or (horizon, num_samples).
    The returned array has length min(horizon, len(observed)).
    """
    x = np.asarray(samples, dtype=np.float64)
    y = np.asarray(observed, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or y.size == 0:
        return np.array([], dtype=np.float64)

    if x.shape[0] == y.size and x.shape[1] != y.size:
        sample_matrix = x.T
    else:
        sample_matrix = x

    horizon = min(sample_matrix.shape[1], y.size)
    sample_matrix = sample_matrix[:, :horizon]
    y = y[:horizon]
    return np.mean(sample_matrix <= y[None, :], axis=0)


def empirical_cdf_values_from_quantiles(
    quantiles: np.ndarray,
    quantile_levels: np.ndarray,
    observed: np.ndarray,
) -> np.ndarray:
    """Approximate F(observed_i | H_{i-1}) by interpolating model quantiles."""
    q = np.asarray(quantiles, dtype=np.float64)
    levels = np.asarray(quantile_levels, dtype=np.float64).reshape(-1)
    y = np.asarray(observed, dtype=np.float64).reshape(-1)
    if q.ndim != 2 or levels.size == 0 or y.size == 0:
        return np.array([], dtype=np.float64)

    if q.shape[1] == levels.size:
        q_by_horizon = q
    elif q.shape[0] == levels.size:
        q_by_horizon = q.T
    else:
        return np.array([], dtype=np.float64)

    order = np.argsort(levels)
    levels = levels[order]
    q_by_horizon = q_by_horizon[:, order]

    horizon = min(q_by_horizon.shape[0], y.size)
    z = []
    for i in range(horizon):
        qi = q_by_horizon[i]
        finite = np.isfinite(qi) & np.isfinite(levels)
        if finite.sum() < 2:
            z.append(math.nan)
            continue
        value_order = np.argsort(qi[finite])
        z.append(
            float(
                np.interp(
                    y[i],
                    qi[finite][value_order],
                    levels[finite][value_order],
                    left=0.0,
                    right=1.0,
                )
            )
        )
    return np.asarray(z, dtype=np.float64)


def _waveform_rr_pit_from_waveform_quantiles(
    record: Dict[str, Any],
    result: Dict[str, Any],
    lead_index: int,
    truth: np.ndarray,
    *,
    context_length: int,
    sampling_rate: float,
) -> np.ndarray:
    quantiles = _first_present(result, ["quantiles", "forecast_quantiles"])
    levels = _first_present(result, ["quantile_levels", "quantile_probs"])
    if quantiles is None or levels is None:
        return np.array([], dtype=np.float64)

    waveform_quantiles = _lead_array(quantiles, lead_index)
    quantile_levels = np.asarray(levels, dtype=np.float64).reshape(-1)
    if waveform_quantiles.ndim != 2 or quantile_levels.size == 0:
        return np.array([], dtype=np.float64)

    if waveform_quantiles.shape[-1] == quantile_levels.size:
        waveform_by_quantile = np.moveaxis(waveform_quantiles, -1, 0)
    elif waveform_quantiles.shape[0] == quantile_levels.size:
        waveform_by_quantile = waveform_quantiles
    else:
        return np.array([], dtype=np.float64)

    max_rr = len(truth)
    if max_rr == 0:
        return np.array([], dtype=np.float64)

    rr_quantiles = np.full((max_rr, len(quantile_levels)), np.nan, dtype=np.float64)
    context_signal = record["signals"][lead_index, :context_length]
    for quantile_index, predicted_signal in enumerate(waveform_by_quantile):
        rr = extract_rr_from_waveform(
            context_signal,
            predicted_signal,
            sampling_rate=sampling_rate,
        )
        limit = min(max_rr, len(rr))
        if limit:
            rr_quantiles[:limit, quantile_index] = rr[:limit]

    return empirical_cdf_values_from_quantiles(rr_quantiles, quantile_levels, truth)


def _lead_array(value: Any, lead_index: int) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim >= 2 and arr.shape[0] > lead_index:
        return np.asarray(arr[lead_index])
    return arr


def _first_present(result: Dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in result and result[key] is not None:
            return result[key]
    return None


def _pit_values_from_result(
    result: Dict[str, Any],
    lead_index: int,
    truth: np.ndarray,
    *,
    prefix: str,
) -> np.ndarray:
    """Extract or derive PIT values from a forecast result dictionary.

    Returns an empty array when only point forecasts are available. A true
    time-rescaling/PIT KS statistic requires a conditional predictive
    distribution, not just a point estimate.
    """
    direct = _first_present(
        result,
        [
            f"{prefix}_pit",
            f"{prefix}_pits",
            f"{prefix}_z",
            f"{prefix}_cdf",
            f"{prefix}_cdf_values",
        ],
    )
    if direct is not None:
        return _lead_array(direct, lead_index).astype(np.float64).reshape(-1)

    samples = _first_present(
        result,
        [
            f"{prefix}_samples",
            f"{prefix}_sample",
            f"{prefix}_forecast_samples",
        ],
    )
    if samples is not None:
        return empirical_cdf_values_from_samples(_lead_array(samples, lead_index), truth)

    quantiles = _first_present(
        result,
        [
            f"{prefix}_quantiles",
            f"{prefix}_forecast_quantiles",
        ],
    )
    levels = _first_present(
        result,
        [
            f"{prefix}_quantile_levels",
            f"{prefix}_quantile_probs",
            "quantile_levels",
            "quantile_probs",
        ],
    )
    if quantiles is not None and levels is not None:
        return empirical_cdf_values_from_quantiles(_lead_array(quantiles, lead_index), levels, truth)

    return np.array([], dtype=np.float64)


def aggregate_point_metrics(pairs: Sequence[tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
    errors = []
    window_rmse = []
    window_mae = []
    for truth, pred in pairs:
        limit = min(len(truth), len(pred))
        if limit == 0:
            continue
        truth_limited = np.asarray(truth[:limit], dtype=np.float32)
        pred_limited = np.asarray(pred[:limit], dtype=np.float32)
        diff = pred_limited - truth_limited
        errors.append(diff)
        window_rmse.append(float(np.sqrt(np.mean(diff**2))))
        window_mae.append(float(np.mean(np.abs(diff))))

    output = _empty_metric_dict()
    if not errors:
        return output

    all_errors = np.concatenate(errors)
    output.update(
        {
            "rmse": float(np.sqrt(np.mean(all_errors**2))),
            "mae": float(np.mean(np.abs(all_errors))),
            "window_rmse_mean": float(np.mean(window_rmse)),
            "window_mae_mean": float(np.mean(window_mae)),
        }
    )
    return output


def aggregate_pit_ks(pit_pairs: Sequence[tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
    """Aggregate one-sample PIT KS scores across windows/leads."""
    ks_values = []
    cutoffs = []
    passes = []
    for _truth, pit_values in pit_pairs:
        result = pit_ks_from_cdf_values(pit_values)
        if not math.isnan(result["ks"]):
            ks_values.append(result["ks"])
            cutoffs.append(result["ks_cutoff"])
            passes.append(result["ks_pass"])

    output = _empty_metric_dict()
    output.update(
        {
            "ks_mean": float(np.mean(ks_values)) if ks_values else math.nan,
            "ks_cutoff_mean": float(np.mean(cutoffs)) if cutoffs else math.nan,
            "ks_pass_rate": float(np.mean(passes)) if passes else math.nan,
        }
    )
    return output


def merge_distribution_metrics(point_metrics: Dict[str, Any], ks_metrics: Dict[str, Any]) -> Dict[str, Any]:
    output = dict(point_metrics)
    output["ks_mean"] = ks_metrics.get("ks_mean", math.nan)
    output["ks_cutoff_mean"] = ks_metrics.get("ks_cutoff_mean", math.nan)
    output["ks_pass_rate"] = ks_metrics.get("ks_pass_rate", math.nan)
    return output

def aggregate_metrics(pairs: Sequence[tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
    return aggregate_point_metrics(pairs)

def step_metrics(pairs: Sequence[tuple[np.ndarray, np.ndarray]], horizon: int) -> List[Dict[str, Any]]:
    rows = []
    for step_index in range(int(horizon)):
        diffs = []
        for truth, pred in pairs:
            if step_index < len(truth) and step_index < len(pred):
                diffs.append(float(pred[step_index] - truth[step_index]))
        if diffs:
            arr = np.asarray(diffs, dtype=np.float32)
            rows.append(
                {
                    "step_index": int(step_index),
                    "rmse": float(np.sqrt(np.mean(arr**2))),
                    "mae": float(np.mean(np.abs(arr))),
                    "count": int(len(arr)),
                }
            )
        else:
            rows.append({"step_index": int(step_index), "rmse": math.nan, "mae": math.nan, "count": 0})
    return rows


def metric_row(
    *,
    model_name: str,
    context_length: int,
    horizon: int,
    rr_context: int,
    rr_horizon: int,
    records: Sequence[Dict[str, Any]],
    waveform_metrics: Dict[str, Any],
    direct_rr_metrics: Dict[str, Any],
    extracted_rr_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "sweep_type": "waveform_context",
        "model": model_name,
        "context_length": int(context_length),
        "horizon": int(horizon),
        "rr_context_beats": int(rr_context),
        "rr_horizon_beats": int(rr_horizon),
        "rmse": waveform_metrics["rmse"],
        "mae": waveform_metrics["mae"],
        "window_rmse_mean": waveform_metrics["window_rmse_mean"],
        "window_mae_mean": waveform_metrics["window_mae_mean"],
        "num_windows": int(len(records)),
        "num_series": int(sum(len(record["lead_names"]) for record in records)),
        "rr_rmse": direct_rr_metrics["rmse"],
        "rr_mae": direct_rr_metrics["mae"],
        "rr_window_rmse_mean": direct_rr_metrics["window_rmse_mean"],
        "rr_window_mae_mean": direct_rr_metrics["window_mae_mean"],
        "rr_ks_mean": direct_rr_metrics["ks_mean"],
        "rr_ks_cutoff_mean": direct_rr_metrics.get("ks_cutoff_mean", math.nan),
        "rr_ks_pass_rate": direct_rr_metrics.get("ks_pass_rate", math.nan),
        "waveform_rr_rmse": extracted_rr_metrics["rmse"],
        "waveform_rr_mae": extracted_rr_metrics["mae"],
        "waveform_rr_window_rmse_mean": extracted_rr_metrics["window_rmse_mean"],
        "waveform_rr_window_mae_mean": extracted_rr_metrics["window_mae_mean"],
        "waveform_rr_ks_mean": extracted_rr_metrics["ks_mean"],
        "waveform_rr_ks_cutoff_mean": extracted_rr_metrics.get("ks_cutoff_mean", math.nan),
        "waveform_rr_ks_pass_rate": extracted_rr_metrics.get("ks_pass_rate", math.nan),
    }


def generate_paper_metrics(
    *,
    model_keys: Sequence[str],
    data_root: Path | None = None,
    record_ids: Sequence[str] = ("all",),
    contexts: Sequence[int] = (2048, 4096, 8192),
    horizons: Sequence[int] = (256, 512, 1024),
    rr_context: int | None = None,
    rr_horizon: int | None = None,
    max_windows: int = 32,
    metrics_path: Path = Path("figures/paper/rr_sweep.csv"),
    step_metrics_path: Path = Path("figures/paper/rr_sweep_step_metrics.csv"),
    results_dir: Path = Path("figures/results"),
    moirai2_device: str = "cpu",
    moirai2_batch_size: int = 32,
) -> None:
    resolved_data_root = _resolve_default_data_root("nsrdb") if data_root is None else Path(data_root)
    subject_ids = resolve_subject_ids(record_ids)
    all_subject_metric_rows: List[Dict[str, Any]] = []
    all_subject_step_rows: List[Dict[str, Any]] = []
    model_names = [MODEL_NAMES[key] for key in model_keys]
    results_dir = Path(results_dir)

    max_context = max(int(value) for value in contexts)
    print(f"Running paper metrics for {len(subject_ids)} subject(s): {', '.join(subject_ids)}")
    print(f"Subject cache directory: {results_dir}")

    for subject_index, subject_id in enumerate(subject_ids, start=1):
        metric_cache_path = _cache_path(results_dir, subject_id, "rr_sweep")
        step_cache_path = _cache_path(results_dir, subject_id, "rr_sweep_step_metrics")
        subject_metric_rows: List[Dict[str, Any]] = list(read_rows(metric_cache_path))
        subject_step_rows: List[Dict[str, Any]] = list(read_rows(step_cache_path))

        print(
            f"\nSubject {subject_index}/{len(subject_ids)}: {subject_id} "
            f"({len(subject_metric_rows)} cached metric rows)"
        )

        for horizon in horizons:
            missing_for_horizon = []
            for context_length in contexts:
                current_rr_context = int(context_length) if rr_context is None else int(rr_context)
                current_rr_horizon = int(horizon) if rr_horizon is None else int(rr_horizon)
                cached_keys = {
                    _row_key(row, METRIC_KEY_FIELDS)
                    for row in subject_metric_rows
                    if _metric_row_has_ks(row)
                }
                for model_key in model_keys:
                    model_name = MODEL_NAMES[model_key]
                    key = _metric_cache_key(
                        model_name,
                        int(context_length),
                        int(horizon),
                        current_rr_context,
                        current_rr_horizon,
                    )
                    if key not in cached_keys:
                        missing_for_horizon.append(
                            (
                                int(context_length),
                                int(horizon),
                                current_rr_context,
                                current_rr_horizon,
                                model_key,
                                model_name,
                            )
                        )

            if not missing_for_horizon:
                print(f"  Horizon {horizon}: loaded from cache")
                continue

            base_records = build_evaluation_records(
                resolved_data_root,
                [subject_id],
                max_context,
                int(horizon),
                max_windows,
            )
            if not base_records:
                raise RuntimeError(f"No evaluation records were available for subject={subject_id}, horizon={horizon}.")

            for (
                context_length,
                current_horizon,
                current_rr_context,
                current_rr_horizon,
                model_key,
                model_name,
            ) in missing_for_horizon:
                print(
                    f"  Subject {subject_index}/{len(subject_ids)} {subject_id}: "
                    f"{model_name}, context={context_length}, horizon={current_horizon}, "
                    f"rr_context={current_rr_context}, rr_horizon={current_rr_horizon}, "
                    f"windows={len(base_records)}"
                )
                results = forecast_records(
                    base_records,
                    current_horizon,
                    context_length,
                    current_rr_context,
                    current_rr_horizon,
                    model_key,
                    moirai2_device=moirai2_device,
                    moirai2_batch_size=moirai2_batch_size,
                )
                waveform_metrics = aggregate_point_metrics(
                    waveform_forecast_pairs(base_records, results, context_length, current_horizon)
                )

                direct_pairs = direct_rr_pairs(base_records, results, context_length, current_rr_horizon)
                direct_point_metrics = aggregate_point_metrics(direct_pairs)
                direct_ks_metrics = aggregate_pit_ks(
                    direct_rr_pit_pairs(base_records, results, context_length, current_rr_horizon)
                )
                direct_metrics = merge_distribution_metrics(direct_point_metrics, direct_ks_metrics)

                extracted_pairs = waveform_pairs(base_records, results, context_length, current_horizon)
                extracted_point_metrics = aggregate_point_metrics(extracted_pairs)
                extracted_ks_metrics = aggregate_pit_ks(
                    waveform_rr_pit_pairs(base_records, results, context_length, current_horizon)
                )
                extracted_metrics = merge_distribution_metrics(extracted_point_metrics, extracted_ks_metrics)
                metric_key = _metric_cache_key(
                    model_name,
                    context_length,
                    current_horizon,
                    current_rr_context,
                    current_rr_horizon,
                )
                subject_metric_rows = [
                    row for row in subject_metric_rows if _row_key(row, METRIC_KEY_FIELDS) != metric_key
                ]
                subject_step_rows = [
                    row for row in subject_step_rows if _row_key(row, METRIC_KEY_FIELDS) != metric_key
                ]
                subject_metric_rows.append(
                    {
                        "subject_id": subject_id,
                        **metric_row(
                            model_name=model_name,
                            context_length=context_length,
                            horizon=current_horizon,
                            rr_context=current_rr_context,
                            rr_horizon=current_rr_horizon,
                            records=base_records,
                            waveform_metrics=waveform_metrics,
                            direct_rr_metrics=direct_metrics,
                            extracted_rr_metrics=extracted_metrics,
                        ),
                    }
                )
                for step_row in step_metrics(direct_pairs, current_rr_horizon):
                    subject_step_rows.append(
                        {
                            "subject_id": subject_id,
                            "sweep_type": "waveform_context",
                            "metric_type": "rr",
                            "model": model_name,
                            "context_length": context_length,
                            "horizon": current_horizon,
                            "rr_context_beats": int(current_rr_context),
                            "rr_horizon_beats": int(current_rr_horizon),
                            **step_row,
                        }
                    )

                subject_metric_rows = _sort_metric_rows(subject_metric_rows)
                subject_step_rows = _sort_step_rows(subject_step_rows)
                write_rows(metric_cache_path, subject_metric_rows, SUBJECT_METRIC_FIELDNAMES)
                write_rows(step_cache_path, subject_step_rows, SUBJECT_STEP_FIELDNAMES)
                print(f"    Cached subject results to {metric_cache_path}")

        all_subject_metric_rows.extend(subject_metric_rows)
        all_subject_step_rows.extend(subject_step_rows)

    expected_metric_keys = {
        _metric_cache_key(
            MODEL_NAMES[model_key],
            int(context_length),
            int(horizon),
            int(context_length) if rr_context is None else int(rr_context),
            int(horizon) if rr_horizon is None else int(rr_horizon),
        )
        for horizon in horizons
        for context_length in contexts
        for model_key in model_keys
    }
    selected_subject_metric_rows = [
        row
        for row in all_subject_metric_rows
        if row.get("model") in set(model_names)
        and _row_key(row, METRIC_KEY_FIELDS) in expected_metric_keys
    ]
    selected_subject_step_rows = [
        row
        for row in all_subject_step_rows
        if row.get("model") in set(model_names)
        and _row_key(row, METRIC_KEY_FIELDS) in expected_metric_keys
    ]
    metrics_rows = _average_subject_rows(
        selected_subject_metric_rows,
        key_fields=METRIC_KEY_FIELDS,
        fieldnames=METRIC_FIELDNAMES,
    )
    step_rows = _average_subject_rows(
        selected_subject_step_rows,
        key_fields=STEP_KEY_FIELDS,
        fieldnames=STEP_FIELDNAMES,
    )

    merge_model_rows(metrics_path, metrics_rows, model_names, METRIC_FIELDNAMES)
    merge_model_rows(step_metrics_path, step_rows, model_names, STEP_FIELDNAMES)
    print(
        f"Averaged {len(subject_ids)} subject cache(s) into {metrics_path} "
        f"and {step_metrics_path}"
    )
