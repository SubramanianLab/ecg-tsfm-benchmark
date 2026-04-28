"""Plotting helpers for ECG and RR forecasts."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def _mask_outliers(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    masked = np.asarray(values, dtype=np.float32).copy()
    masked[(masked < lower) | (masked > upper)] = np.nan
    return masked


def _outlier_bounds(series: List[np.ndarray], multiplier: float = 3.0) -> tuple[float, float]:
    values = np.concatenate([np.asarray(item, dtype=np.float32).reshape(-1) for item in series if len(item)])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return -np.inf, np.inf
    q1, q3 = np.percentile(values, [25, 75])
    iqr = float(q3 - q1)
    if iqr <= 1e-8:
        return -np.inf, np.inf
    return float(q1 - multiplier * iqr), float(q3 + multiplier * iqr)


def plot_forecasts(
    records: List[Dict[str, Any]],
    results_by_model: List[List[Dict[str, Any]]],
    output: str,
    rr_context_beats: int,
    rr_horizon_beats: int,
) -> None:
    n_rows = sum(len(record["lead_names"]) for record in records)
    n_models = max(len(results_by_model), 1)
    record_labels = [f"{record['record_id']}({','.join(record['lead_names'])})" for record in records]
    fig, axes = plt.subplots(
        n_rows * 2,
        n_models,
        figsize=(5.8 * n_models, max(4.6 * n_rows, 4.8)),
        squeeze=False,
    )
    fig.suptitle(f"MIT-BIH Forecasts: {'; '.join(record_labels)}", fontsize=12)
    row = 0
    for record_index, record in enumerate(records):
        for lead_index, lead in enumerate(record["lead_names"]):
            signal = record["signals"][lead_index]
            sampling_rate = record["sampling_rate"]
            context_length = record.get("context_length", 0)
            horizon = record.get("horizon", len(signal) - context_length)
            context_signal = signal[:context_length]
            gt_future_signal = signal[context_length : context_length + horizon]
            context_time = np.arange(context_length) / sampling_rate
            future_time = np.arange(context_length, context_length + len(gt_future_signal)) / sampling_rate
            gt_rpeaks = np.asarray(record.get("annotation_samples", []), dtype=np.int64)
            full_rpeaks_abs = np.asarray(record.get("full_annotation_samples_abs", []), dtype=np.int64)
            forecast_boundary_abs = int(record["sample_offset"]) + context_length
            context_rpeaks = gt_rpeaks[gt_rpeaks < context_length]
            future_rpeaks = gt_rpeaks[(gt_rpeaks >= context_length) & (gt_rpeaks < context_length + horizon)]
            full_rr_intervals = (
                np.diff(full_rpeaks_abs).astype(np.float32)
                if len(full_rpeaks_abs) > 1
                else np.array([], dtype=np.float32)
            )
            boundary_peak_idx = int(np.searchsorted(full_rpeaks_abs, forecast_boundary_abs, side="left"))
            boundary_rr_idx = max(0, boundary_peak_idx - 1)
            rr_context_start = max(0, boundary_rr_idx - rr_context_beats)
            context_rr = full_rr_intervals[rr_context_start:boundary_rr_idx]
            gt_future_rr = full_rr_intervals[boundary_rr_idx : boundary_rr_idx + rr_horizon_beats]
            context_rr_x = np.arange(rr_context_start, rr_context_start + len(context_rr))
            future_rr_x = np.arange(boundary_rr_idx, boundary_rr_idx + len(gt_future_rr))
            for col, model_results in enumerate(results_by_model):
                result = model_results[record_index]
                ax_wave = axes[row * 2][col]
                ax_rr = axes[row * 2 + 1][col]
                predicted_waveform = result["point"][lead_index]
                predicted_rr = result["rr_point"][lead_index][: len(gt_future_rr)]
                ax_wave.plot(context_time, context_signal, color="0.35", linewidth=0.9, label="Context waveform")
                ax_wave.plot(
                    future_time,
                    gt_future_signal,
                    color="black",
                    linestyle="--",
                    linewidth=1.1,
                    label="Ground truth future",
                )
                ax_wave.plot(
                    future_time,
                    predicted_waveform[: len(future_time)],
                    color=result["color"],
                    linewidth=1.1,
                    label=f"{result['name']} forecast",
                )
                if len(context_rpeaks) > 0:
                    ax_wave.scatter(
                        context_rpeaks / sampling_rate,
                        context_signal[context_rpeaks],
                        color="0.45",
                        s=10,
                        zorder=5,
                        label="Context R-peaks",
                    )
                if len(future_rpeaks) > 0:
                    future_peak_idx = future_rpeaks - context_length
                    ax_wave.scatter(
                        future_rpeaks / sampling_rate,
                        gt_future_signal[future_peak_idx],
                        color="black",
                        s=11,
                        zorder=6,
                        label="Ground truth future R-peaks",
                    )
                ax_wave.axvline(context_length / sampling_rate, color="0.6", linestyle=":", linewidth=0.9)
                ax_wave.set_title(f"{result['name']} Waveform - {record['record_id']} {lead}", fontsize=9)
                ax_wave.set_xlabel("Time (s)")
                ax_wave.set_ylabel("mV")
                wave_handles, wave_labels = ax_wave.get_legend_handles_labels()
                if wave_handles:
                    ax_wave.legend(wave_handles, wave_labels, fontsize=6, loc="upper right", frameon=False)
                ax_wave.grid(True, alpha=0.3)
                ax_wave.spines["top"].set_visible(False)
                ax_wave.spines["right"].set_visible(False)
                ax_rr.plot(context_rr_x, context_rr, color="0.45", linewidth=0.8, label="Context RR intervals")
                ax_rr.plot(
                    future_rr_x,
                    gt_future_rr,
                    color="black",
                    linestyle="--",
                    linewidth=1.0,
                    label="Ground truth future RR",
                )
                ax_rr.plot(
                    future_rr_x,
                    predicted_rr,
                    color=result["color"],
                    linewidth=1.0,
                    label=f"{result['name']} predicted RR",
                )
                ax_rr.axvline(boundary_rr_idx, color="0.6", linestyle=":", linewidth=0.9)
                ax_rr.set_title(f"{result['name']} RR - {record['record_id']} {lead}", fontsize=9)
                ax_rr.set_xlabel("RR interval index")
                ax_rr.set_ylabel("RR interval (samples)")
                handles, labels = ax_rr.get_legend_handles_labels()
                if handles:
                    ax_rr.legend(
                        handles,
                        labels,
                        fontsize=6,
                        loc="upper center",
                        bbox_to_anchor=(0.5, -0.24),
                        ncol=2,
                        frameon=False,
                    )
                ax_rr.grid(True, alpha=0.3)
                ax_rr.spines["top"].set_visible(False)
                ax_rr.spines["right"].set_visible(False)
            row += 1
    fig.subplots_adjust(hspace=0.42)
    plt.tight_layout(rect=(0, 0.04, 1, 0.98))
    plt.savefig(output, dpi=150)
    print(f"\nPlot saved to {output}")
    plt.show()

def plot_publication_rr_forecasts(
    records: List[Dict[str, Any]],
    results_by_model: List[List[Dict[str, Any]]],
    output: str,
    rr_context_beats: int,
    rr_horizon_beats: int,
) -> None:
    n_lead_rows = sum(len(record["lead_names"]) for record in records)
    n_rows = n_lead_rows * 2
    n_cols = max(len(results_by_model), 1)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.8 * n_cols, max(1.8 * n_rows + 1.0, 4.2)),
        squeeze=False,
    )
    legend_map: Dict[str, Any] = {}

    row = 0
    for record_index, record in enumerate(records):
        if "rr_context" not in record or "rr_future" not in record or "rr_boundary_index" not in record:
            raise ValueError(
                "Publication RR figure requires exact RR windows on each record. "
                "Prepare records with rr_context/rr_future/rr_boundary_index first."
        )
        boundary_rr_idx = int(record["rr_boundary_index"])
        sampling_rate = float(record["sampling_rate"])
        context_length = int(record.get("context_length", 0))
        horizon = int(record.get("horizon", 0))
        context_rr = np.asarray(record["rr_context"], dtype=np.float32)
        gt_future_rr = np.asarray(record["rr_future"], dtype=np.float32)
        context_rr_x = np.arange(boundary_rr_idx - len(context_rr), boundary_rr_idx)
        future_rr_x = np.arange(boundary_rr_idx, boundary_rr_idx + len(gt_future_rr))

        for lead_index, lead in enumerate(record["lead_names"]):
            signal = np.asarray(record["signals"][lead_index], dtype=np.float32)
            context_signal = signal[:context_length]
            future_signal = signal[context_length : context_length + horizon]
            context_time = np.arange(len(context_signal)) / sampling_rate
            future_time = np.arange(context_length, context_length + len(future_signal)) / sampling_rate
            predicted_by_model = []
            for model_results in results_by_model:
                result = model_results[record_index]
                predicted_rr = np.asarray(result["rr_point"][lead_index], dtype=np.float32)[: len(gt_future_rr)]
                predicted_by_model.append((result, predicted_rr))
            lower, upper = _outlier_bounds(
                [context_rr / sampling_rate, gt_future_rr / sampling_rate]
                + [predicted / sampling_rate for _, predicted in predicted_by_model]
            )
            for col, model_results in enumerate(results_by_model):
                ax_wave = axes[row][col]
                ax_rr = axes[row + 1][col]
                result = model_results[record_index]
                predicted_waveform = np.asarray(result["point"][lead_index], dtype=np.float32)[: len(future_signal)]
                predicted_rr = np.asarray(result["rr_point"][lead_index], dtype=np.float32)[: len(gt_future_rr)]
                ax_wave.plot(
                    context_time,
                    context_signal,
                    color="0.55",
                    linewidth=0.8,
                    label="Context Signal",
                )
                ax_wave.plot(
                    future_time,
                    future_signal,
                    color="0.20",
                    linestyle="--",
                    linewidth=0.95,
                    label="Ground Truth Signal",
                )
                ax_wave.plot(
                    future_time[: len(predicted_waveform)],
                    predicted_waveform,
                    color=result["color"],
                    linewidth=0.95,
                    label=f"{result['name']} Forecast",
                )
                ax_wave.axvline(context_length / sampling_rate, color="0.75", linestyle=":", linewidth=0.8)
                ax_wave.set_title(f"{result['name']} - {lead}", fontsize=9)
                ax_wave.set_xlabel("Time (s)")
                ax_wave.set_ylabel("mV")
                ax_wave.grid(True, alpha=0.22)
                ax_wave.spines["top"].set_visible(False)
                ax_wave.spines["right"].set_visible(False)
                handles, labels = ax_wave.get_legend_handles_labels()
                for handle, label in zip(handles, labels):
                    if label not in legend_map:
                        legend_map[label] = handle

                ax_rr.plot(
                    context_rr_x,
                    _mask_outliers(context_rr / sampling_rate, lower, upper),
                    color="0.70",
                    linewidth=0.9,
                    label="Context Signal",
                )
                ax_rr.plot(
                    future_rr_x,
                    _mask_outliers(gt_future_rr / sampling_rate, lower, upper),
                    color="0.20",
                    linestyle="--",
                    linewidth=1.0,
                    label="Ground Truth Signal",
                )
                ax_rr.plot(
                    future_rr_x,
                    _mask_outliers(predicted_rr / sampling_rate, lower, upper),
                    color=result["color"],
                    linewidth=1.0,
                    label=f"{result['name']} Forecast",
                )
                ax_rr.axvline(boundary_rr_idx, color="0.75", linestyle=":", linewidth=0.8)
                ax_rr.set_xlabel("RR interval index")
                ax_rr.set_ylabel("RR interval (s)")
                ax_rr.grid(True, alpha=0.22)
                ax_rr.spines["top"].set_visible(False)
                ax_rr.spines["right"].set_visible(False)
                handles, labels = ax_rr.get_legend_handles_labels()
                for handle, label in zip(handles, labels):
                    if label not in legend_map:
                        legend_map[label] = handle

            row += 2

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    if legend_map:
        fig.legend(
            list(legend_map.values()),
            list(legend_map.keys()),
            loc="upper center",
            ncol=len(legend_map),
            fontsize=7,
            frameon=False,
            bbox_to_anchor=(0.5, 0.96),
        )
    plt.tight_layout(rect=(0, 0, 1, 0.935))
    plt.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Saved publication RR figure to {output}")
