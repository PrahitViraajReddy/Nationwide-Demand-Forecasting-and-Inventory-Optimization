"""Forecast evaluation metrics with explicit zero-denominator handling."""

from __future__ import annotations

import numpy as np


def wape(actual, forecast) -> float:
    """Weighted absolute percentage error, returned as a ratio (0.12 = 12%)."""
    y = np.asarray(actual, dtype=float).reshape(-1)
    yhat = np.asarray(forecast, dtype=float).reshape(-1)
    if y.shape != yhat.shape:
        raise ValueError("actual and forecast must have the same shape")
    if not np.isfinite(y).all() or not np.isfinite(yhat).all():
        raise ValueError("actual and forecast must contain only finite values")
    denominator = np.abs(y).sum()
    if denominator == 0:
        raise ValueError("WAPE is undefined when all actual values are zero")
    return float(np.abs(y - yhat).sum() / denominator)


def rmsse(actual, forecast, training_actual, seasonal_period: int = 1) -> float:
    """Root mean squared scaled error using training-history scaling.

    seasonal_period=1 scales against one-step naive differences. Use 7 for a
    weekly seasonal-naive scale on daily retail series when appropriate.
    """
    y = np.asarray(actual, dtype=float).reshape(-1)
    yhat = np.asarray(forecast, dtype=float).reshape(-1)
    train = np.asarray(training_actual, dtype=float).reshape(-1)
    if y.shape != yhat.shape:
        raise ValueError("actual and forecast must have the same shape")
    if not np.isfinite(y).all() or not np.isfinite(yhat).all() or not np.isfinite(train).all():
        raise ValueError("metric inputs must contain only finite values")
    if seasonal_period < 1 or len(train) <= seasonal_period:
        raise ValueError("training_actual must be longer than seasonal_period >= 1")
    scale = np.mean((train[seasonal_period:] - train[:-seasonal_period]) ** 2)
    if scale <= 0:
        raise ValueError("RMSSE is undefined because the training scale is zero")
    return float(np.sqrt(np.mean((y - yhat) ** 2) / scale))
