# Evaluting Long-Range Temporal Structure in Foundation Model-Based Forecasts of Heartbeat Dynamics

This repository accompanies the manuscript <i>Evaluating Long-Range Temporal Structure in Foundation Model-Based Forecasts of Heartbeat Dynamics</i>.

## Overview

This project evaluates time-series foundation models on long-range ECG waveform and heartbeat-dynamics forecasting tasks using MIT-BIH WFDB records. It supports waveform forecasting, direct RR-interval forecasting from beat annotations, and RR extraction from generated waveforms.

## Repository Contents

| Path | Description |
| --- | --- |
| `main.py` | Primary command-line entry point for ECG forecasting, plotting, and table/figure generation. |
| `generate_paper_metrics.py` | Runs the manuscript RR metric sweep across models, context lengths, and horizons. |
| `ecg_forecast.py` | CLI parser and top-level workflow dispatch. |
| `ecg_workflows.py` | Shared forecast preparation and model execution workflow. |
| `ecg_dataloader.py` | MIT-BIH/WFDB record loading, metadata parsing, lead selection, and annotation loading. |
| `ecg_rr.py` | R-peak and RR-interval helper utilities. |
| `models.py` | Wrappers for TimesFM, Chronos-2, and Moirai 2.0 inference. |
| `ecg_paper_metrics.py` | Metric aggregation for waveform forecasts, direct RR forecasts, and waveform-derived RR intervals. |
| `ecg_plots.py` | Forecast and RR publication plotting utilities. |
| `visualization.py` | Manuscript table and per-beat RR metric figure generation from CSV outputs. |
| `figures/paper/` | Existing manuscript artifacts, including figures, metric CSVs, and table outputs. |

## Models

The benchmark currently includes:

- **TimesFM**: `google/timesfm-2.5-200m-pytorch`
- **Chronos-2**: `amazon/chronos-2`
- **Moirai 2.0**: `Salesforce/moirai-2.0-R-small`

The `--model all` option runs all three models. The `--model both` option runs the older TimesFM plus Chronos-2 pair.

## Data

The loaders expect MIT-BIH records in WFDB format. By default, dataset roots are resolved one directory above this repository:

- Arrhythmia database: `../mit-bih-arrhythmia-database-1.0.0`
- Normal sinus rhythm database: `../mit-bih-normal-sinus-rhythm-database-1.0.0`

If those paths are not present, the code also checks the archived layout under `../Archive/`.

You can override these locations with `--data-root`.

Examples:

```bash
python main.py --dataset nsrdb --list-records
python main.py --dataset arrhythmia --data-root /path/to/mit-bih-arrhythmia-database-1.0.0 --list-records
```

## Installation

This repository does not currently include a pinned environment file. A typical environment needs Python 3.10+ and the packages used by the benchmark:

```bash
pip install numpy scipy matplotlib pandas torch wfdb
pip install timesfm chronos-forecasting uni2ts gluonts
```

Model downloads are handled by the corresponding model libraries and may require Hugging Face access/network connectivity on first use. Moirai 2.0 defaults to CPU because the Uni2TS/GluonTS path can emit tensors that are incompatible with MPS in this workflow.

## Basic Forecasting

Run the default NSRDB example:

```bash
python main.py
```

Run one NSRDB record with a specific model, context, and horizon:

```bash
python main.py 16265 \
  --dataset nsrdb \
  --model timesfm \
  --waveform-context 8192 \
  --waveform-horizon 1024 \
  --rr-context 2000 \
  --rr-horizon 256 \
  --output figures/nsrdb/record_16265_timesfm.png
```

Run all available leads for an arrhythmia record:

```bash
python main.py 100 \
  --dataset arrhythmia \
  --all-leads \
  --model all \
  --waveform-context 8192 \
  --waveform-horizon 1024
```

## Manuscript Metrics

Generate the RR metric sweep used for the manuscript:

```bash
python generate_paper_metrics.py \
  --models all \
  --records all \
  --contexts 2048,4096,8192 \
  --horizons 256,512,1024 \
  --max-windows 32
```

By default, this evaluates the 18-subject MIT-BIH NSRDB cohort, caches per-subject results under `figures/results/`, and writes subject-averaged outputs to:

- `figures/paper/rr_sweep.csv`
- `figures/paper/rr_sweep_step_metrics.csv`

Each subject cache is written incrementally after a model/context/horizon combination completes. If a run is interrupted, rerunning the same command reuses matching cached rows and computes only missing combinations.

You can run a subset of models while preserving existing rows for other models:

```bash
python generate_paper_metrics.py --models timesfm --max-windows 32
```

## Manuscript Figures and Tables

Generate the publication-style RR forecast figure:

```bash
python main.py 16265 \
  --dataset nsrdb \
  --model all \
  --paper-rr-figure \
  --output figures/paper/figure1.png
```

Generate the per-beat RR error figure from step metrics:

```bash
python main.py \
  --paper-step-figure-from-csv figures/paper/rr_sweep_step_metrics.csv \
  --paper-step-figure-output figures/paper/figure2.png \
  --paper-step-contexts 512,2048
```

Generate the manuscript RR comparison table:

```bash
python main.py \
  --paper-table-from-csv figures/paper/rr_sweep.csv \
  --paper-table-output figures/paper/table2_rr \
  --paper-table-contexts 2048,4096,8192 \
  --paper-table-horizons 256,512,1024
```

This produces both:

- `figures/paper/table2_rr.csv`
- `figures/paper/table2_rr.tex`

## Metrics

The evaluation includes:

- Waveform forecast RMSE and MAE.
- Direct RR-interval forecast RMSE, MAE, and Kolmogorov-Smirnov statistic.
- RR metrics extracted from forecast waveforms using peak detection.
- Per-step RR RMSE and MAE across the forecast horizon.

RR values are represented in samples during metric generation and converted to seconds in manuscript tables/figures using the configured sampling rate, which defaults to 128 Hz.

## Notes

- The code uses WFDB annotations and filters to heartbeat annotation symbols before constructing RR intervals.
- Forecast windows are built from fixed waveform context and horizon lengths.
- For short RR contexts, the CLI enforces a minimum RR context of 512 beats.
- Existing manuscript outputs are available under `figures/paper/` for reference and regeneration.
