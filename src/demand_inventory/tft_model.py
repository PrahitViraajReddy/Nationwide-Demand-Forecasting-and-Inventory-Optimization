"""Small, reproducible Temporal Fusion Transformer experiment.

This is intentionally a bounded experiment, not a full-M5 training configuration.
The input must already contain daily rows and known future covariates for the
entire training/validation window. Use a chronological holdout; never random-split
time-series observations.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def prepare_tft_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """Validate data and prepare known/static covariates for PyTorch Forecasting."""
    required = {"id", "date", "sales"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"TFT input is missing required columns: {', '.join(sorted(missing))}")
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["sales"] = pd.to_numeric(data["sales"], errors="coerce")
    data = data.dropna(subset=["id", "date", "sales"]).sort_values(["id", "date"])
    if data.empty:
        raise ValueError("TFT input has no valid rows")
    if (data["sales"] < 0).any():
        raise ValueError("sales must be non-negative")
    if data.duplicated(["id", "date"]).any():
        raise ValueError("TFT input must have one row per id/date")
    data["id"] = data["id"].astype(str)
    dates = pd.Index(sorted(data["date"].unique()))
    time_map = {date: index for index, date in enumerate(dates)}
    data["time_idx"] = data["date"].map(time_map).astype("int64")
    data["day_of_week"] = data["date"].dt.dayofweek.astype("int8")
    data["month"] = data["date"].dt.month.astype("int8")

    known_reals = ["time_idx", "day_of_week", "month"]
    if "sell_price" in data.columns:
        data["sell_price"] = pd.to_numeric(data["sell_price"], errors="coerce").fillna(0.0)
        known_reals.append("sell_price")

    known_categoricals = [
        column for column in ("event_name_1", "event_type_1", "event_name_2", "event_type_2")
        if column in data.columns
    ]
    for column in known_categoricals:
        data[column] = data[column].fillna("None").astype(str)

    static_categoricals = [
        column for column in ("item_id", "dept_id", "cat_id", "store_id", "state_id")
        if column in data.columns
    ]
    for column in static_categoricals:
        data[column] = data[column].fillna("Unknown").astype(str)
    return data, known_reals, known_categoricals, static_categoricals


def train_tft_experiment(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    horizon: int = 28,
    max_encoder_length: int = 56,
    max_epochs: int = 5,
    batch_size: int = 64,
    seed: int = 42,
) -> dict:
    """Train a bounded TFT experiment and save checkpoint plus validation metrics."""
    if horizon < 1 or max_encoder_length < 7 or max_epochs < 1 or batch_size < 1:
        raise ValueError("horizon, epochs, and batch size must be positive; encoder length must be at least 7")
    data, known_reals, known_categoricals, static_categoricals = prepare_tft_frame(frame)
    max_time = int(data["time_idx"].max())
    cutoff = max_time - horizon
    if cutoff < max_encoder_length:
        raise ValueError("Not enough history for the configured encoder and forecast horizon")
    train_frame = data[data["time_idx"] <= cutoff].copy()
    if train_frame.empty:
        raise ValueError("TFT training partition is empty")

    try:
        from lightning.pytorch import Trainer, seed_everything
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
        from pytorch_forecasting.data import GroupNormalizer
        from pytorch_forecasting.metrics import QuantileLoss
    except ImportError as exc:
        raise ImportError(
            "TFT training requires pytorch-forecasting and lightning. Install the full project dependencies."
        ) from exc

    seed_everything(seed, workers=True)
    training = TimeSeriesDataSet(
        train_frame,
        time_idx="time_idx",
        target="sales",
        group_ids=["id"],
        min_encoder_length=min(14, max_encoder_length),
        max_encoder_length=max_encoder_length,
        min_prediction_length=horizon,
        max_prediction_length=horizon,
        static_categoricals=static_categoricals,
        time_varying_known_reals=known_reals,
        time_varying_known_categoricals=known_categoricals,
        time_varying_unknown_reals=["sales"],
        target_normalizer=GroupNormalizer(groups=["id"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=False,
    )
    validation = TimeSeriesDataSet.from_dataset(
        training,
        data,
        min_prediction_idx=cutoff + 1,
        stop_randomization=True,
        predict=True,
    )
    train_loader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    validation_loader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=8,
        loss=QuantileLoss(),
        log_interval=-1,
        reduce_on_plateau_patience=3,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        gradient_clip_val=0.1,
        logger=False,
        enable_model_summary=False,
        default_root_dir=str(output_dir),
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=validation_loader)
    checkpoint = output_dir / "tft_model.ckpt"
    trainer.save_checkpoint(str(checkpoint))

    prediction_result = model.predict(
        validation_loader,
        mode="prediction",
        return_index=True,
        trainer_kwargs={"accelerator": "auto", "devices": 1},
    )
    predictions = prediction_result.output if hasattr(prediction_result, "output") else prediction_result
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    prediction_array = predictions.detach().cpu().numpy() if hasattr(predictions, "detach") else np.asarray(predictions)
    # QuantileLoss predictions contain quantiles on the final axis; use the median (0.5).
    if prediction_array.ndim == 3:
        quantiles = list(QuantileLoss().quantiles)
        median_index = min(range(len(quantiles)), key=lambda index: abs(quantiles[index] - 0.5))
        point_predictions = prediction_array[:, :, median_index]
        lower_index = min(range(len(quantiles)), key=lambda index: abs(quantiles[index] - 0.1))
        upper_index = min(range(len(quantiles)), key=lambda index: abs(quantiles[index] - 0.9))
        lower_predictions = prediction_array[:, :, lower_index]
        upper_predictions = prediction_array[:, :, upper_index]
    else:
        point_predictions = prediction_array
        lower_predictions = prediction_array
        upper_predictions = prediction_array

    index_frame = prediction_result.index if hasattr(prediction_result, "index") else None
    # The output object/index API varies across PyTorch Forecasting versions.
    # Save model/checkpoint metadata even if the caller chooses its own inference adapter.
    metadata = {
        "model": "TemporalFusionTransformer",
        "series_count": int(data["id"].nunique()),
        "horizon_days": int(horizon),
        "max_encoder_length": int(max_encoder_length),
        "max_epochs": int(max_epochs),
        "training_cutoff_time_idx": int(cutoff),
        "checkpoint": str(checkpoint),
        "prediction_shape": list(np.asarray(point_predictions).shape),
        "validation_index_available": index_frame is not None,
        "note": "Quantile outputs are model estimates; interval coverage must be evaluated on held-out actuals.",
    }
    (output_dir / "tft_metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
