"""Prophet benchmark with explicit forecast intervals and input validation."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def forecast_with_prophet(
    history: pd.DataFrame,
    *,
    horizon_days: int = 28,
    interval_width: float = 0.95,
    holiday_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Fit a univariate Prophet model and return non-negative forecast intervals.

    Args:
        history: DataFrame containing daily `date` and non-negative `sales`.
        horizon_days: Number of future daily periods to forecast.
        interval_width: Central predictive interval width, between 0 and 1.
        holiday_frame: Optional Prophet holiday table with `ds` and `holiday`.
    """
    if horizon_days < 1:
        raise ValueError("horizon_days must be at least 1")
    if not 0 < interval_width < 1:
        raise ValueError("interval_width must be strictly between 0 and 1")
    if not {"date", "sales"}.issubset(history.columns):
        raise ValueError("history must contain date and sales columns")

    frame = history[["date", "sales"]].copy()
    frame["ds"] = pd.to_datetime(frame.pop("date"), errors="coerce")
    frame["y"] = pd.to_numeric(frame.pop("sales"), errors="coerce")
    frame = frame.dropna(subset=["ds", "y"]).sort_values("ds")
    if frame.empty:
        raise ValueError("history has no valid date/sales observations")
    if (frame["y"] < 0).any():
        raise ValueError("sales must be non-negative")
    if frame["ds"].duplicated().any():
        raise ValueError("history must contain at most one observation per date")
    if len(frame) < 14:
        raise ValueError("at least 14 observations are required for the Prophet benchmark")

    # Prophet's interval is conditional on its fitted model and is not a guarantee
    # of empirical coverage; evaluation must measure coverage on held-out dates.
    try:
        from prophet import Prophet
    except ImportError as exc:
        raise ImportError("Prophet is required for this model. Install it with: pip install prophet") from exc

    kwargs = {
        "interval_width": float(interval_width),
        "weekly_seasonality": True,
        "yearly_seasonality": "auto",
        "daily_seasonality": False,
    }
    if holiday_frame is not None:
        if not {"ds", "holiday"}.issubset(holiday_frame.columns):
            raise ValueError("holiday_frame must contain ds and holiday columns")
        kwargs["holidays"] = holiday_frame.copy()
    model = Prophet(**kwargs)
    model.fit(frame[["ds", "y"]])
    future = model.make_future_dataframe(periods=int(horizon_days), freq="D", include_history=False)
    predicted = model.predict(future)
    result = predicted[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(
        columns={"ds": "date", "yhat": "forecast", "yhat_lower": "lower", "yhat_upper": "upper"}
    )
    for column in ("forecast", "lower", "upper"):
        result[column] = result[column].clip(lower=0.0)
    result["lower"] = np.minimum(result["lower"], result["forecast"])
    result["upper"] = np.maximum(result["upper"], result["forecast"])
    result["interval_width"] = result["upper"] - result["lower"]
    return result.reset_index(drop=True)


def interval_metrics(actual: Sequence[float], lower: Sequence[float], upper: Sequence[float]) -> dict[str, float]:
    """Return empirical prediction-interval coverage and mean interval width."""
    y = np.asarray(actual, dtype=float).reshape(-1)
    lo = np.asarray(lower, dtype=float).reshape(-1)
    hi = np.asarray(upper, dtype=float).reshape(-1)
    if y.shape != lo.shape or y.shape != hi.shape:
        raise ValueError("actual, lower, and upper must have the same shape")
    if y.size == 0:
        raise ValueError("interval metric inputs cannot be empty")
    if not np.isfinite(y).all() or not np.isfinite(lo).all() or not np.isfinite(hi).all():
        raise ValueError("interval metric inputs must be finite")
    if (lo > hi).any():
        raise ValueError("lower bounds cannot exceed upper bounds")
    return {
        "coverage": float(np.mean((y >= lo) & (y <= hi))),
        "mean_interval_width": float(np.mean(hi - lo)),
    }
