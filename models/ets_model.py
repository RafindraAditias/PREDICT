# # models/ets_model.py
# import pandas as pd
# import numpy as np
# from statsmodels.tsa.holtwinters import ExponentialSmoothing
# from utils.metrics import calc_metrics

# def run_ets_model(
#     series: pd.Series,
#     horizon: int = 30,
#     test_size: int = 102,
#     trend="add",
#     seasonal="add",
#     seasonal_periods=30,
#     initialization_method="estimated"
    
# ):
#     s = series.dropna().astype(float)
#     s = s[~s.index.duplicated(keep="last")]
#     s = s.sort_index()

#     if len(s) <= test_size + 10:
#         raise ValueError("Data terlalu sedikit untuk split train test. Kurangi test_size atau tambah data.")

#     train = s.iloc[:-test_size]
#     test = s.iloc[-test_size:]

#     model = ExponentialSmoothing(
#         train,
#         trend=trend,
#         seasonal=seasonal,
#         seasonal_periods=int(seasonal_periods),
#         initialization_method=initialization_method
#     )
#     fit = model.fit(optimized=True)

#     pred_test = fit.forecast(len(test))
#     pred_test = pd.Series(np.asarray(pred_test), index=test.index, name="test_pred")

#     m = calc_metrics(test.values, pred_test.values)
#     m["Model"] = f"ETS(trend={trend}, seasonal={seasonal}, s={int(seasonal_periods)})"
#     m["TestSize"] = int(test_size)

#     model_full = ExponentialSmoothing(
#         s,
#         trend=trend,
#         seasonal=seasonal,
#         seasonal_periods=int(seasonal_periods),
#         initialization_method=initialization_method
#     )
#     fit_full = model_full.fit(optimized=True)

#     future = fit_full.forecast(horizon)
#     future_index = pd.date_range(start=s.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
#     future = pd.Series(np.asarray(future), index=future_index, name="forecast")

#     return {
#         "forecast": future,
#         "test_pred": pred_test,
#         "train": train,
#         "test": test,
#         "metrics": m,
#         "params": {"trend": trend, "seasonal": seasonal, "seasonal_periods": int(seasonal_periods)},
#     }

# models/ets_model.py
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from utils.metrics import calc_metrics

# ============================================================
# PARAMETER HASIL OPTIMASI (hardcoded untuk reproduktibilitas)
# Diperoleh dari fitting pada data lokal dengan optimized=True
# Ganti nilai di bawah dengan hasil print(fit.params) dari lokal
# ============================================================
_TRAIN_PARAMS = {
    "smoothing_level":    None,   # alpha  <-- ISI DARI LOKAL
    "smoothing_trend":    None,   # beta   <-- ISI DARI LOKAL
    "smoothing_seasonal": None,   # gamma  <-- ISI DARI LOKAL
}

_FULL_PARAMS = {
    "smoothing_level":    None,   # alpha  <-- ISI DARI LOKAL
    "smoothing_trend":    None,   # beta   <-- ISI DARI LOKAL
    "smoothing_seasonal": None,   # gamma  <-- ISI DARI LOKAL
}

# ============================================================
# HELPER: Print params (jalankan sekali di lokal lalu matikan)
# ============================================================
_DEBUG_PRINT_PARAMS = True   # set False setelah dapat nilainya


def run_ets_model(
    series: pd.Series,
    horizon: int = 30,
    test_size: int = 102,
    trend="add",
    seasonal="add",
    seasonal_periods=30,
    initialization_method="estimated",
):
    s = series.dropna().astype(float)
    s = s[~s.index.duplicated(keep="last")]
    s = s.sort_index()

    if len(s) <= test_size + 10:
        raise ValueError(
            "Data terlalu sedikit untuk split train test. Kurangi test_size atau tambah data."
        )

    train = s.iloc[:-test_size]
    test = s.iloc[-test_size:]

    # ----------------------------------------------------------
    # FIT 1: pada data TRAIN saja (untuk menghitung metrik)
    # ----------------------------------------------------------
    model_train = ExponentialSmoothing(
        train,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=int(seasonal_periods),
        initialization_method=initialization_method,
    )

    use_hardcode_train = all(v is not None for v in _TRAIN_PARAMS.values())

    if use_hardcode_train:
        fit = model_train.fit(
            optimized=False,
            smoothing_level=_TRAIN_PARAMS["smoothing_level"],
            smoothing_trend=_TRAIN_PARAMS["smoothing_trend"],
            smoothing_seasonal=_TRAIN_PARAMS["smoothing_seasonal"],
        )
    else:
        fit = model_train.fit(optimized=True)
        if _DEBUG_PRINT_PARAMS:
            print("=" * 50)
            print("TRAIN MODEL PARAMS (copy ke _TRAIN_PARAMS):")
            print(f"  smoothing_level:    {fit.params['smoothing_level']}")
            print(f"  smoothing_trend:    {fit.params['smoothing_trend']}")
            print(f"  smoothing_seasonal: {fit.params['smoothing_seasonal']}")
            print("=" * 50)

    pred_test = fit.forecast(len(test))
    pred_test = pd.Series(np.asarray(pred_test), index=test.index, name="test_pred")

    m = calc_metrics(test.values, pred_test.values)
    m["Model"] = f"ETS(trend={trend}, seasonal={seasonal}, s={int(seasonal_periods)})"
    m["TestSize"] = int(test_size)

    # ----------------------------------------------------------
    # FIT 2: pada data FULL (untuk forecast ke depan)
    # ----------------------------------------------------------
    model_full = ExponentialSmoothing(
        s,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=int(seasonal_periods),
        initialization_method=initialization_method,
    )

    use_hardcode_full = all(v is not None for v in _FULL_PARAMS.values())

    if use_hardcode_full:
        fit_full = model_full.fit(
            optimized=False,
            smoothing_level=_FULL_PARAMS["smoothing_level"],
            smoothing_trend=_FULL_PARAMS["smoothing_trend"],
            smoothing_seasonal=_FULL_PARAMS["smoothing_seasonal"],
        )
    else:
        fit_full = model_full.fit(optimized=True)
        if _DEBUG_PRINT_PARAMS:
            print("=" * 50)
            print("FULL MODEL PARAMS (copy ke _FULL_PARAMS):")
            print(f"  smoothing_level:    {fit_full.params['smoothing_level']}")
            print(f"  smoothing_trend:    {fit_full.params['smoothing_trend']}")
            print(f"  smoothing_seasonal: {fit_full.params['smoothing_seasonal']}")
            print("=" * 50)

    future = fit_full.forecast(horizon)
    future_index = pd.date_range(
        start=s.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D"
    )
    future = pd.Series(np.asarray(future), index=future_index, name="forecast")

    return {
        "forecast": future,
        "test_pred": pred_test,
        "train": train,
        "test": test,
        "metrics": m,
        "params": {
            "trend": trend,
            "seasonal": seasonal,
            "seasonal_periods": int(seasonal_periods),
        },
    }