"""Public CLI entry point for ECG forecasting."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from ecg_config import (
    DEFAULT_RR_CONTEXT_BEATS,
    MIN_RR_CONTEXT_BEATS,
    _default_output_name,
    _parse_int_list,
    _require_input_file,
    _resolve_default_data_root,
    _resolve_input_path,
    _resolve_output_path,
)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Forecast MIT-BIH ECG windows from arrhythmia or NSRDB with TimesFM, Chronos-2, and/or Moirai 2.0"
    )
    parser.add_argument("records", nargs="*", help="MIT-BIH record ids or WFDB stem paths")
    parser.add_argument(
        "--dataset",
        type=str,
        default="nsrdb",
        choices=["arrhythmia", "nsrdb"],
        help="Which bundled MIT-BIH dataset to use: arrhythmia or normal sinus rhythm database (nsrdb)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Dataset root override; defaults to the selected bundled dataset",
    )
    parser.add_argument("--list-records", action="store_true", help="List available records and exit")
    parser.add_argument(
        "--leads",
        nargs="+",
        default=None,
        help="Lead names to forecast, or use 'all'; defaults to all available leads",
    )
    parser.add_argument("--all-leads", action="store_true", help="Forecast all available leads for each record")
    parser.add_argument("--start-sample", type=int, default=0, help="Window start sample within the record")
    parser.add_argument(
        "--waveform-context",
        "--context-length",
        dest="waveform_context",
        type=int,
        default=8192,
        help="Waveform context length in samples",
    )
    parser.add_argument(
        "--waveform-horizon",
        "--horizon",
        dest="waveform_horizon",
        type=int,
        default=1024,
        help="Waveform forecast horizon in samples",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["timesfm", "chronos", "moirai2", "both", "all"],
        help="'all' runs TimesFM, Chronos-2, and Moirai 2.0; 'both' keeps the older TimesFM+Chronos-2 pair",
    )
    parser.add_argument(
        "--moirai2-device",
        type=str,
        default="cpu",
        help="Torch device for Moirai 2.0. CPU is the default; use cuda explicitly with a small --moirai2-batch-size.",
    )
    parser.add_argument(
        "--moirai2-batch-size",
        type=int,
        default=32,
        help="Batch size for Moirai 2.0 inference. For CUDA, 1 or 2 is safer with long ECG contexts.",
    )
    parser.add_argument("--normalize", action="store_true", help="Z-score normalize each selected lead")
    parser.add_argument(
        "--rr-context",
        "--rr-context-beats",
        dest="rr_context",
        type=int,
        default=DEFAULT_RR_CONTEXT_BEATS,
        help="RR-interval context length in beats",
    )
    parser.add_argument(
        "--rr-horizon",
        "--rr-horizon-beats",
        dest="rr_horizon",
        type=int,
        default=256,
        help="RR-interval forecast horizon in beats",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output plot filename; defaults to figures/{dataset}/mit_bih_{dataset}_{sample}.png",
    )
    parser.add_argument(
        "--paper-rr-figure",
        action="store_true",
        help="Render an RR-only publication-style figure for the selected records/models.",
    )
    parser.add_argument(
        "--paper-step-figure-from-csv",
        type=str,
        default=None,
        help="Build the per-beat RR figure from a step-metrics CSV.",
    )
    parser.add_argument(
        "--paper-step-figure-output",
        type=str,
        default=None,
        help="Output path for the per-beat publication figure.",
    )
    parser.add_argument(
        "--paper-step-contexts",
        type=str,
        default="512,2048",
        help="Comma-separated context lengths for the per-beat publication figure.",
    )
    parser.add_argument(
        "--paper-step-horizon",
        type=int,
        default=None,
        help="Optional horizon filter for the per-beat publication figure.",
    )
    parser.add_argument(
        "--paper-step-cohort-label",
        type=str,
        default="",
        help="Figure-level label for the per-beat publication figure.",
    )
    parser.add_argument(
        "--paper-step-metric-type",
        type=str,
        default="rr",
        choices=["rr", "rr_variability", "waveform_rr_variability"],
        help="Step-metric family to plot for the publication figure.",
    )
    parser.add_argument(
        "--paper-step-sampling-rate",
        type=float,
        default=128.0,
        help="Sampling rate in Hz used to convert per-beat RR errors from samples to seconds.",
    )
    parser.add_argument(
        "--paper-table-from-csv",
        type=str,
        default=None,
        help="Build the publication RR comparison table from an evaluation CSV.",
    )
    parser.add_argument(
        "--paper-table-output",
        type=str,
        default=None,
        help="Output prefix/path for the publication table. Produces .tex and .csv companions.",
    )
    parser.add_argument(
        "--paper-table-contexts",
        type=str,
        default="2048,4096,8192",
        help="Comma-separated context lengths for the publication table.",
    )
    parser.add_argument(
        "--paper-table-horizons",
        type=str,
        default="256,512,1024",
        help="Comma-separated horizons for the publication table.",
    )
    parser.add_argument(
        "--paper-table-sampling-rate",
        type=float,
        default=128.0,
        help="Sampling rate in Hz used to convert table RR errors from samples to seconds.",
    )
    return parser
def run_cli(args: argparse.Namespace) -> None:
    default_records = {"arrhythmia": ["100"], "nsrdb": ["16265"]}
    if not args.records:
        args.records = default_records[args.dataset]
    if args.data_root is None:
        args.data_root = str(_resolve_default_data_root(args.dataset))
    if args.output is None and args.paper_rr_figure:
        args.output = str(_resolve_output_path("figures/paper/figure1.png"))
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    elif args.output is None:
        args.output = _default_output_name(args.dataset, args.records)
    else:
        args.output = str(_resolve_output_path(args.output))
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    from ecg_dataloader import list_mitbih_records

    if args.list_records:
        for record_id in list_mitbih_records(args.data_root):
            print(record_id)
        return
    if args.paper_step_figure_from_csv:
        from visualization import plot_publication_rr_step_figure

        step_csv_path = _resolve_input_path(args.paper_step_figure_from_csv)
        _require_input_file(step_csv_path, "Step-metrics CSV")
        output_path = _resolve_output_path(
            args.paper_step_figure_output
            or step_csv_path.with_name("figure2.png")
        )
        plot_publication_rr_step_figure(
            step_csv_path,
            output_path,
            contexts=_parse_int_list(args.paper_step_contexts),
            horizon=args.paper_step_horizon,
            metric_type=args.paper_step_metric_type,
            cohort_label=args.paper_step_cohort_label,
            sampling_rate_hz=args.paper_step_sampling_rate,
        )
        return
    if args.paper_table_from_csv:
        from visualization import write_publication_rr_table

        csv_path = _resolve_input_path(args.paper_table_from_csv)
        _require_input_file(csv_path, "Evaluation CSV")
        output_path = _resolve_output_path(
            args.paper_table_output or csv_path.with_name(csv_path.stem + "_paper_table")
        )
        write_publication_rr_table(
            csv_path,
            output_path,
            contexts=_parse_int_list(args.paper_table_contexts),
            horizons=_parse_int_list(args.paper_table_horizons),
            sampling_rate_hz=args.paper_table_sampling_rate,
        )
        return
    from ecg_plots import plot_forecasts, plot_publication_rr_forecasts
    from ecg_workflows import forecast_records, prepare_forecast_window

    if args.rr_context < MIN_RR_CONTEXT_BEATS:
        print(f"Adjusting RR context to {MIN_RR_CONTEXT_BEATS} intervals.")
        args.rr_context = MIN_RR_CONTEXT_BEATS
    args.requested_leads = "all" if args.all_leads or args.leads is None or args.leads == ["all"] else args.leads
    print(f"Loading {args.dataset.upper()} records {args.records} from {args.data_root}")
    records = []
    for record_id in args.records:
        args.record = record_id
        record = prepare_forecast_window(args)
        records.append(record)
        print(f"  Record: {record['record_id']}")
        print(f"    Sampling rate: {record['sampling_rate']} Hz")
        print(f"    Available leads: {record['available_leads']}")
        print(f"    Selected leads: {record['lead_names']}")
        print(f"    Patient age/sex: {record['patient_age']} / {record['patient_sex']}")
        print(f"    Window offset: {record['sample_offset']}")
        print(f"    Annotation count: {len(record.get('annotation_symbols', []))}")
    results_by_model: List[List[Dict[str, Any]]] = []
    if args.model in {"timesfm", "both", "all"}:
        results_by_model.append(
            forecast_records(
                records,
                args.waveform_horizon,
                args.waveform_context,
                args.rr_context,
                args.rr_horizon,
                "timesfm",
            )
        )
    if args.model in {"chronos", "both", "all"}:
        results_by_model.append(
            forecast_records(
                records,
                args.waveform_horizon,
                args.waveform_context,
                args.rr_context,
                args.rr_horizon,
                "chronos",
            )
        )
    if args.model in {"moirai2", "all"}:
        results_by_model.append(
            forecast_records(
                records,
                args.waveform_horizon,
                args.waveform_context,
                args.rr_context,
                args.rr_horizon,
                "moirai2",
                moirai2_device=args.moirai2_device,
                moirai2_batch_size=args.moirai2_batch_size,
            )
        )
    if args.paper_rr_figure:
        plot_publication_rr_forecasts(
            records,
            results_by_model,
            args.output,
            args.rr_context,
            args.rr_horizon,
        )
        return

    plot_forecasts(records, results_by_model, args.output, args.rr_context, args.rr_horizon)
