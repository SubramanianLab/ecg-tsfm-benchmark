from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np


_INT_FIELDS = {
    "context_length",
    "horizon",
    "rr_context_beats",
    "rr_horizon_beats",
    "num_windows",
    "num_series",
    "step_index",
    "count",
}

_FLOAT_FIELDS = {
    "rmse",
    "mae",
    "window_rmse_mean",
    "window_mae_mean",
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
}

MODEL_ORDER = ("Chronos-2", "TimesFM", "Moirai 2.0")
MODEL_COLORS = {
    "Chronos-2": "crimson",
    "TimesFM": "royalblue",
    "Moirai 2.0": "darkgreen",
}
MODEL_SLUGS = {
    "Chronos-2": "chronos",
    "TimesFM": "timesfm",
    "Moirai 2.0": "moirai2",
}


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "nan", "none", "null", "na", "n/a"}:
        return True
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def _csv_value(value: Any) -> Any:
    return "" if _is_missing_value(value) else value


def load_evaluation_rows(csv_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed: Dict[str, Any] = {}
            for key, value in row.items():
                if value is None:
                    parsed[key] = value
                elif key in _INT_FIELDS:
                    parsed[key] = 0 if _is_missing_value(value) else int(value)
                elif key in _FLOAT_FIELDS:
                    parsed[key] = math.nan if _is_missing_value(value) else float(value)
                else:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def _metric_or_dash(value: Any, decimals: int = 2) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    if math.isnan(numeric):
        return "—"
    return f"{numeric:.{decimals}f}"


def _samples_to_seconds_or_none(value: Any, sampling_rate_hz: float) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric / float(sampling_rate_hz)


def _rolling_median(values: Sequence[float], window: int = 21) -> List[float]:
    arr = [float(value) for value in values]
    if window <= 1 or len(arr) < 3:
        return arr
    radius = max(1, int(window) // 2)
    smoothed: List[float] = []
    for index in range(len(arr)):
        start = max(0, index - radius)
        stop = min(len(arr), index + radius + 1)
        smoothed.append(float(np.median(arr[start:stop])))
    return smoothed


def plot_publication_rr_step_figure(
    step_csv_path: Path,
    output_path: Path,
    *,
    contexts: Sequence[int],
    horizon: int | None = None,
    metric_type: str = "rr",
    cohort_label: str = "",
    sampling_rate_hz: float = 128.0,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_label = "RR Intervals" if metric_type.lower() == "rr" else metric_type
    rows = load_evaluation_rows(step_csv_path)
    filtered = [
        row
        for row in rows
        if row.get("sweep_type") == "waveform_context"
        and row.get("metric_type") == metric_type
        and int(row.get("context_length", -1)) in {int(value) for value in contexts}
        and (horizon is None or int(row.get("horizon", -1)) == int(horizon))
    ]
    if not filtered:
        raise ValueError("No matching step-metric rows found for publication figure.")

    ordered_contexts = [int(value) for value in contexts]
    fig, axes = plt.subplots(
        len(ordered_contexts),
        2,
        figsize=(12.5, max(3.2 * len(ordered_contexts), 4.2)),
        squeeze=False,
    )
    has_cohort_label = bool(cohort_label.strip())
    if has_cohort_label:
        fig.suptitle(cohort_label, fontsize=13, y=0.995)
    for row_index, context_length in enumerate(ordered_contexts):
        context_rows = [row for row in filtered if int(row["context_length"]) == context_length]
        for col_index, metric_name in enumerate(("rmse", "mae")):
            axis = axes[row_index][col_index]
            for model_name in MODEL_ORDER:
                model_rows = sorted(
                    [row for row in context_rows if str(row["model"]) == model_name],
                    key=lambda row: int(row["step_index"]),
                )
                if not model_rows:
                    continue
                axis.plot(
                    [int(row["step_index"]) for row in model_rows],
                    _rolling_median(
                        [float(row[metric_name]) / float(sampling_rate_hz) for row in model_rows]
                    ),
                    linewidth=1.7,
                    color=MODEL_COLORS.get(model_name),
                    label=model_name,
                )
            axis.set_title(f"{metric_label} {metric_name.upper()} at context={context_length}")
            axis.set_xlabel("Forecast beat index")
            axis.set_ylabel(f"RR interval {metric_name.upper()} (s)")
            axis.grid(True, alpha=0.25)
            axis.legend(fontsize=8, frameon=False)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=(0, 0, 1, 0.96 if has_cohort_label else 1))
    plt.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved publication step figure to {output_path}")


def write_publication_rr_table(
    csv_path: Path,
    output_path: Path,
    *,
    contexts: Sequence[int],
    horizons: Sequence[int],
    sampling_rate_hz: float = 128.0,
) -> None:
    rows = load_evaluation_rows(csv_path)
    row_lookup = {
        (int(row["context_length"]), int(row["horizon"]), str(row["model"])): row
        for row in rows
        if row.get("sweep_type") == "waveform_context"
    }

    csv_output = output_path.with_suffix(".csv")
    tex_output = output_path.with_suffix(".tex")
    csv_output.parent.mkdir(parents=True, exist_ok=True)

    csv_rows: List[Dict[str, Any]] = []
    for context_length in contexts:
        for horizon in horizons:
            for output_type in ("Direct", "Extracted"):
                for model_name in MODEL_ORDER:
                    model_row = row_lookup.get((int(context_length), int(horizon), model_name), {})
                    if output_type == "Direct":
                        rmse = _samples_to_seconds_or_none(model_row.get("rr_rmse"), sampling_rate_hz)
                        mae = _samples_to_seconds_or_none(model_row.get("rr_mae"), sampling_rate_hz)
                        ks = model_row.get("rr_ks_mean")
                    else:
                        rmse = _samples_to_seconds_or_none(model_row.get("waveform_rr_rmse"), sampling_rate_hz)
                        mae = _samples_to_seconds_or_none(model_row.get("waveform_rr_mae"), sampling_rate_hz)
                        ks = model_row.get("waveform_rr_ks_mean")
                    csv_rows.append(
                        {
                            "context_length": int(context_length),
                            "prediction_horizon": int(horizon),
                            "output_type": output_type,
                            "model": model_name,
                            "rmse_s": rmse,
                            "mae_s": mae,
                            "ks": ks,
                        }
                    )

    with csv_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(
            [{key: _csv_value(value) for key, value in row.items()} for row in csv_rows]
        )

    lines: List[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Forecasting performance across context lengths and prediction horizons.}")
    lines.append(r"\label{tab:forecast_results}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\begin{tabular}{ll ll ccc}")
    lines.append(r"\toprule")
    lines.append(
        r"Context Length & Prediction Horizon & Output Type & Model "
        r"& RMSE (s) $\downarrow$ & MAE (s) $\downarrow$ & KS $\downarrow$ \\"
    )
    lines.append(r"\midrule")
    lines.append("")

    ordered_contexts = [int(value) for value in contexts]
    ordered_horizons = [int(value) for value in horizons]
    for context_index, context_length in enumerate(ordered_contexts):
        for horizon_index, horizon in enumerate(ordered_horizons):
            group_rows: List[List[str]] = []
            for output_type in ("Direct", "Extracted"):
                for model_name in MODEL_ORDER:
                    model_row = row_lookup.get((context_length, horizon, model_name), {})
                    if output_type == "Direct":
                        rmse = _samples_to_seconds_or_none(model_row.get("rr_rmse"), sampling_rate_hz)
                        mae = _samples_to_seconds_or_none(model_row.get("rr_mae"), sampling_rate_hz)
                        ks = model_row.get("rr_ks_mean")
                    else:
                        rmse = _samples_to_seconds_or_none(model_row.get("waveform_rr_rmse"), sampling_rate_hz)
                        mae = _samples_to_seconds_or_none(model_row.get("waveform_rr_mae"), sampling_rate_hz)
                        ks = model_row.get("waveform_rr_ks_mean")
                    group_rows.append(
                        [
                            "",
                            "",
                            "",
                            model_name,
                            _metric_or_dash(rmse),
                            _metric_or_dash(mae),
                            _metric_or_dash(ks),
                        ]
                    )
            group_rows[0][0] = rf"\multirow{{{len(group_rows)}}}{{*}}{{{context_length:,}}}"
            group_rows[0][1] = rf"\multirow{{{len(group_rows)}}}{{*}}{{{horizon:,}}}"
            group_rows[0][2] = rf"\multirow{{{len(MODEL_ORDER)}}}{{*}}{{Direct}}"
            group_rows[len(MODEL_ORDER)][2] = rf"\multirow{{{len(MODEL_ORDER)}}}{{*}}{{Extracted}}"
            for row_index, row_values in enumerate(group_rows):
                if row_index == len(MODEL_ORDER):
                    lines.append(r"\cmidrule(lr){3-7}")
                lines.append(" & ".join(row_values) + r" \\")
            is_last_horizon = horizon_index == len(ordered_horizons) - 1
            if not is_last_horizon:
                lines.append(r"\cmidrule(lr){1-7}")
                lines.append("")
        if context_index != len(ordered_contexts) - 1:
            lines.append("")
            lines.append(r"\midrule")
            lines.append("")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\vskip -0.1in")
    lines.append(r"\end{table*}")
    tex_output.write_text("\n".join(lines) + "\n")
    print(f"Saved publication table to {tex_output}")
    print(f"Saved publication table data to {csv_output}")
