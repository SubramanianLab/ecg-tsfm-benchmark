from __future__ import annotations

import argparse
from pathlib import Path

from ecg_config import _parse_int_list
from ecg_paper_metrics import generate_paper_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate paper RR metrics with the shared TimesFM/Chronos-2/Moirai2 evaluation path."
    )
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Comma-separated model keys to run: timesfm, chronos, moirai2, or all.",
    )
    parser.add_argument(
        "--records",
        type=str,
        default="all",
        help="Comma-separated MIT-BIH NSRDB record ids, or all for the 18-subject NSRDB cohort.",
    )
    parser.add_argument("--contexts", type=str, default="2048,4096,8192", help="Comma-separated waveform contexts.")
    parser.add_argument("--horizons", type=str, default="256,512,1024", help="Comma-separated waveform horizons.")
    parser.add_argument(
        "--rr-context",
        type=int,
        default=None,
        help="Optional RR context override. Defaults to each waveform context length.",
    )
    parser.add_argument(
        "--rr-horizon",
        type=int,
        default=None,
        help="Optional RR horizon override. Defaults to each prediction horizon.",
    )
    parser.add_argument("--max-windows", type=int, default=32, help="Maximum windows to evaluate.")
    parser.add_argument("--data-root", type=Path, default=None, help="Dataset root override.")
    parser.add_argument("--metrics-output", type=Path, default=Path("figures/paper/rr_sweep.csv"))
    parser.add_argument(
        "--step-metrics-output",
        type=Path,
        default=Path("figures/paper/rr_sweep_step_metrics.csv"),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("figures/results"),
        help="Directory for per-subject cached metric CSVs used to resume interrupted runs.",
    )
    parser.add_argument(
        "--moirai2-device",
        type=str,
        default="cuda",
        help="Torch device for Moirai 2.0. CPU is the default; use cuda explicitly with a small --moirai2-batch-size.",
    )
    parser.add_argument("--moirai2-batch-size", type=int, default=32)
    return parser


def parse_models(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return ["timesfm", "chronos", "moirai2"]
    models = [item.strip().lower() for item in raw.split(",") if item.strip()]
    allowed = {"timesfm", "chronos", "moirai2"}
    unknown = sorted(set(models) - allowed)
    if unknown:
        raise ValueError(f"Unsupported model key(s): {', '.join(unknown)}")
    return models


def main() -> None:
    args = build_parser().parse_args()
    generate_paper_metrics(
        model_keys=parse_models(args.models),
        data_root=args.data_root,
        record_ids=[record.strip() for record in args.records.split(",") if record.strip()],
        contexts=_parse_int_list(args.contexts),
        horizons=_parse_int_list(args.horizons),
        rr_context=args.rr_context,
        rr_horizon=args.rr_horizon,
        max_windows=args.max_windows,
        metrics_path=args.metrics_output,
        step_metrics_path=args.step_metrics_output,
        results_dir=args.results_dir,
        moirai2_device=args.moirai2_device,
        moirai2_batch_size=args.moirai2_batch_size,
    )


if __name__ == "__main__":
    main()
