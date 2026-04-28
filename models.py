from __future__ import annotations
import inspect
from typing import Any, Dict, List
import numpy as np
import torch

_MOIRAI2_MODULE = None
MOIRAI2_MODEL_ID = "Salesforce/moirai-2.0-R-small"
MOIRAI2_MODEL_NAME = "Moirai 2.0"

def resolve_torch_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def _fixed_length_context(signal: np.ndarray, context_length: int) -> np.ndarray:
    series = np.asarray(signal, dtype=np.float32).reshape(-1)
    if series.size:
        finite = np.isfinite(series)
        if not finite.all():
            fill = float(np.nanmedian(series[finite])) if finite.any() else 0.0
            series = np.where(finite, series, fill).astype(np.float32)
    trimmed = series[-context_length:]
    if trimmed.shape[0] >= context_length:
        return trimmed.astype(np.float32, copy=False)
    padded = np.zeros((context_length,), dtype=np.float32)
    if trimmed.shape[0] > 0:
        padded[:] = float(trimmed[0])
        padded[-trimmed.shape[0] :] = trimmed
    return padded

def load_timesfm_model():
    import timesfm
    model_id = "google/timesfm-2.5-200m-pytorch"
    model_cls = timesfm.TimesFM_2p5_200M_torch
    try:
        return model_cls.from_pretrained(model_id)
    except TypeError as exc:
        if "unexpected keyword argument 'proxies'" not in str(exc):
            raise
        print(
            "[TimesFM] Detected incompatible `from_pretrained()` implementation. "
            "Falling back to the low-level loader..."
        )
        fallback = getattr(model_cls, "_from_pretrained", None)
        if fallback is None:
            raise RuntimeError(
                "Installed TimesFM version is incompatible and does not expose "
                "`_from_pretrained()`. Upgrade `timesfm` and `huggingface_hub`."
            ) from exc
        kwargs: Dict[str, Any] = {
            "model_id": model_id,
            "revision": None,
            "cache_dir": None,
            "force_download": False,
            "local_files_only": False,
            "token": None,
            "config": None,
        }
        supported = inspect.signature(fallback).parameters
        filtered_kwargs = {key: value for key, value in kwargs.items() if key in supported}
        return fallback(**filtered_kwargs)

def run_timesfm(inputs: List[np.ndarray], horizon: int, context_length: int) -> Dict[str, Any]:
    import timesfm
    torch.set_float32_matmul_precision("high")
    device = resolve_torch_device()
    print(f"\n[TimesFM] Loading model (device={device})...")
    model = load_timesfm_model()
    model_to = getattr(model, "to", None)
    if callable(model_to):
        try:
            model = model_to(device)
        except Exception as exc:
            print(f"[TimesFM] Warning: could not move model to {device}: {exc}")
    model.compile(
        timesfm.ForecastConfig(
            max_context=context_length,
            max_horizon=horizon,
            normalize_inputs=True,
            use_continuous_quantile_head=False,
            force_flip_invariance=True,
            infer_is_positive=False,
            fix_quantile_crossing=False,
        )
    )
    prepared_inputs = [_fixed_length_context(signal, context_length) for signal in inputs]
    print(f"[TimesFM] Forecasting {horizon} samples...")
    point_forecast, _ = model.forecast(horizon=horizon, inputs=prepared_inputs)
    point_forecast = np.asarray(point_forecast, dtype=np.float32)
    if not np.isfinite(point_forecast).all():
        bad_count = int(point_forecast.size - np.isfinite(point_forecast).sum())
        raise RuntimeError(
            f"[TimesFM] produced {bad_count} non-finite forecast values "
            f"for context={context_length}, horizon={horizon}."
        )
    return {
        "name": "TimesFM",
        "color": "royalblue",
        "point": point_forecast,
    }

def run_chronos(inputs: List[np.ndarray], horizon: int, context_length: int) -> Dict[str, Any]:
    from chronos import BaseChronosPipeline, Chronos2Pipeline
    device = resolve_torch_device()
    print(f"\n[Chronos-2] Loading model (device={device})...")
    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-2",
        device_map=device,
    )
    batch = []
    for signal in inputs:
        series = np.asarray(signal, dtype=np.float32).reshape(-1)
        trimmed = series[-context_length:]
        if trimmed.shape[0] < context_length:
            padded = np.zeros((context_length,), dtype=np.float32)
            if trimmed.shape[0] > 0:
                padded[-trimmed.shape[0] :] = trimmed
            trimmed = padded
        batch.append(trimmed.reshape(1, -1))
    batch_np = np.stack(batch, axis=0)
    print(f"[Chronos-2] Forecasting {horizon} samples, input shape={batch_np.shape}...")
    _, mean = pipeline.predict_quantiles(
        batch_np,
        prediction_length=horizon,
        quantile_levels=[0.1, 0.5, 0.9],
    )
    n_leads = len(inputs)
    point_all = np.zeros((n_leads, horizon), dtype=np.float32)
    for index in range(n_leads):
        point_all[index] = mean[index].squeeze(0).cpu().numpy()[:horizon]
    return {
        "name": "Chronos-2",
        "color": "crimson",
        "point": point_all,
    }

def load_moirai2_module():
    global _MOIRAI2_MODULE
    if _MOIRAI2_MODULE is None:
        from uni2ts.model.moirai2 import Moirai2Module
        try:
            _MOIRAI2_MODULE = Moirai2Module.from_pretrained(
                MOIRAI2_MODEL_ID,
                local_files_only=True,
            )
        except Exception:
            _MOIRAI2_MODULE = Moirai2Module.from_pretrained(MOIRAI2_MODEL_ID)
    return _MOIRAI2_MODULE

def run_moirai2(
    inputs: List[np.ndarray],
    horizon: int,
    context_length: int,
    device: str = "cpu",
    batch_size: int = 32,
) -> Dict[str, Any]:
    try:
        import pandas as pd
        from gluonts.dataset.common import ListDataset
        from uni2ts.model.moirai2 import Moirai2Forecast
    except ImportError as exc:
        raise ImportError(
            "Moirai2 support requires uni2ts and GluonTS. Install Uni2TS, then rerun "
            "with --model moirai2 or --model all."
        ) from exc

    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Requested MPS for Moirai2, but torch.backends.mps.is_available() is false.")
    print(f"\n[{MOIRAI2_MODEL_NAME}] Loading model (device={device})...")
    model = Moirai2Forecast(
        module=load_moirai2_module(),
        prediction_length=horizon,
        context_length=context_length,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )
    predictor = model.create_predictor(batch_size=batch_size, device=device)

    dataset_entries = []
    for item_id, signal in enumerate(inputs):
        series = np.asarray(signal, dtype=np.float32).reshape(-1)
        trimmed = series[-context_length:]
        if trimmed.shape[0] < context_length:
            padded = np.zeros((context_length,), dtype=np.float32)
            if trimmed.shape[0] > 0:
                padded[-trimmed.shape[0] :] = trimmed
            trimmed = padded
        dataset_entries.append(
            {
                "start": pd.Period("2000-01-01 00:00:00", freq="s"),
                "target": trimmed,
            }
        )
    dataset = ListDataset(dataset_entries, freq="s")
    print(f"[{MOIRAI2_MODEL_NAME}] Forecasting {horizon} samples...")
    forecasts = list(predictor.predict(dataset))
    point_all = np.zeros((len(inputs), horizon), dtype=np.float32)
    for index, forecast in enumerate(forecasts[: len(inputs)]):
        try:
            point = np.asarray(forecast.mean, dtype=np.float32)
        except Exception:
            point = np.asarray(forecast.quantile("0.5"), dtype=np.float32)
        point_all[index] = point.reshape(-1)[:horizon]
    return {
        "name": MOIRAI2_MODEL_NAME,
        "color": "darkgreen",
        "point": point_all,
    }
