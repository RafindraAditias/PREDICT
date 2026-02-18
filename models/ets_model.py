# models/ets_model.py
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from utils.metrics import calc_metrics

def run_ets_model(
    series: pd.Series,
    horizon: int = 30,
    test_size: int = 102,
    trend="add",
    seasonal="add",
    seasonal_periods=30,
):
    s = series.dropna().astype(float)
    s = s[~s.index.duplicated(keep="last")]
    s = s.sort_index()

    if len(s) <= test_size + 10:
        raise ValueError("Data terlalu sedikit untuk split train test. Kurangi test_size atau tambah data.")

    train = s.iloc[:-test_size]
    test = s.iloc[-test_size:]

    model = ExponentialSmoothing(
        train,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=int(seasonal_periods),
    )
    fit = model.fit(optimized=True)

    pred_test = fit.forecast(len(test))
    pred_test = pd.Series(np.asarray(pred_test), index=test.index, name="test_pred")

    m = calc_metrics(test.values, pred_test.values)
    m["Model"] = f"ETS(trend={trend}, seasonal={seasonal}, s={int(seasonal_periods)})"
    m["TestSize"] = int(test_size)

    model_full = ExponentialSmoothing(
        s,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=int(seasonal_periods),
    )
    fit_full = model_full.fit(optimized=True)

    future = fit_full.forecast(horizon)
    future_index = pd.date_range(start=s.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    future = pd.Series(np.asarray(future), index=future_index, name="forecast")

    return {
        "forecast": future,
        "test_pred": pred_test,
        "train": train,
        "test": test,
        "metrics": m,
        "params": {"trend": trend, "seasonal": seasonal, "seasonal_periods": int(seasonal_periods)},
    }
