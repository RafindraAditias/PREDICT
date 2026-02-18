import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from utils.metrics import calc_metrics

def run_arima_model(
    series: pd.Series,
    horizon: int = 30,
    test_size: int = 90,
    order=(1, 0, 1),
    use_updated_metrics: bool = True,
):
    s = series.dropna().astype(float).sort_index()
    if len(s) <= test_size + 10:
        raise ValueError("Data terlalu sedikit untuk split train test. Kurangi test_size atau tambah data.")

    train = s.iloc[:-test_size]
    test = s.iloc[-test_size:]

    # Fit model pada train
    model = ARIMA(train, order=order)
    fit = model.fit()

    # =========
    # (A) OOS Forecast murni
    # =========
    pred_oos = fit.get_forecast(steps=len(test)).predicted_mean
    pred_oos.index = test.index

    metrics_oos = calc_metrics(test.values, pred_oos.values)
    metrics_oos["Model"] = f"ARIMA{order}"
    metrics_oos["TestSize"] = int(test_size)
    metrics_oos["EvalMode"] = "OOS Forecast"

    # =========
    # (B) Updated Apply — konsisten dengan SARIMA
    # =========
    metrics_updated = None
    fitted_on_test = None
    try:
        updated = fit.apply(test)
        fitted_on_test = pd.Series(
            np.asarray(updated.fittedvalues),
            index=test.index,
            name="fitted_on_test"
        )
        if len(fitted_on_test) != len(test):
            fitted_on_test = fitted_on_test.iloc[-len(test):]
            fitted_on_test.index = test.index

        metrics_updated = calc_metrics(test.values, fitted_on_test.values)
        metrics_updated["Model"] = f"ARIMA{order}"
        metrics_updated["TestSize"] = int(test_size)
        metrics_updated["EvalMode"] = "Updated Apply"

    except Exception as e:
        print(f"Warning: Updated Apply gagal - {e}")
        metrics_updated = None
        fitted_on_test = None

    # =========
    # Tentukan metrik & prediksi utama
    # =========
    if use_updated_metrics and metrics_updated is not None:
        primary_metrics = metrics_updated.copy()
        primary_pred = fitted_on_test
        print(f"✓ Menggunakan Updated Apply metrics - MAPE: {metrics_updated.get('MAPE', 'N/A')}%")
    else:
        primary_metrics = metrics_oos.copy()
        primary_pred = pred_oos
        print(f"✓ Menggunakan OOS Forecast metrics - MAPE: {metrics_oos.get('MAPE', 'N/A')}%")

    # =========
    # Forecast masa depan
    # =========
    model_full = ARIMA(s, order=order)
    fit_full = model_full.fit()

    future = fit_full.get_forecast(steps=horizon).predicted_mean
    future_index = pd.date_range(
        start=s.index[-1] + pd.Timedelta(days=1),
        periods=horizon,
        freq="D"
    )
    future = pd.Series(future.values, index=future_index, name="forecast")

    return {
        "forecast": future,
        "test_pred": primary_pred,
        "train": train,
        "test": test,
        "metrics": primary_metrics,
        "metrics_oos": metrics_oos,
        "metrics_updated": metrics_updated,
        "fitted_on_test": fitted_on_test,
        "params": {"order": order},
    }
# import pandas as pd
# import numpy as np
# from statsmodels.tsa.arima.model import ARIMA
# from utils.metrics import calc_metrics


# def run_arima_model(series: pd.Series,
#                     horizon: int = 30,
#                     test_size: int = 90,
#                     order=(1, 0, 1),
#                     use_updated_metrics=True):
#     """
#     Menjalankan model ARIMA dengan dua mode evaluasi:
#     1. OOS Forecast: evaluasi murni tanpa update
#     2. Rolling Updated: evaluasi dengan update model pada test data
    
#     Parameters:
#     -----------
#     use_updated_metrics : bool, default=True
#         Jika True, gunakan metrik dari Rolling Updated sebagai metrik utama
#         Jika False, gunakan metrik dari OOS Forecast
#     """
#     s = series.dropna().astype(float).sort_index()

#     if len(s) <= test_size + 10:
#         raise ValueError("Data terlalu sedikit untuk split train test. Kurangi test_size atau tambah data.")

#     train = s.iloc[:-test_size]
#     test = s.iloc[-test_size:]

#     # =========================
#     # FIT TRAIN
#     # =========================
#     model = ARIMA(train, order=order)
#     fit = model.fit()

#     # =========================
#     # 1️⃣ Evaluasi Murni (OOS)
#     # =========================
#     pred_oos = fit.get_forecast(steps=len(test)).predicted_mean
#     pred_oos.index = test.index

#     metrics_oos = calc_metrics(test.values, pred_oos.values)
#     metrics_oos["Model"] = f"ARIMA{order}"
#     metrics_oos["TestSize"] = int(len(test))
#     metrics_oos["EvalMode"] = "OOS Forecast"

#     # =========================
#     # 2️⃣ Rolling Updated
#     # =========================
#     history = train.copy()
#     preds_updated = []

#     for t in range(len(test)):
#         model_step = ARIMA(history, order=order)
#         fit_step = model_step.fit()

#         yhat = fit_step.forecast(steps=1)
#         preds_updated.append(yhat.iloc[0])

#         history = pd.concat([history, test.iloc[t:t+1]])

#     pred_updated = pd.Series(preds_updated, index=test.index)

#     metrics_updated = calc_metrics(test.values, pred_updated.values)
#     metrics_updated["Model"] = f"ARIMA{order}"
#     metrics_updated["TestSize"] = int(len(test))
#     metrics_updated["EvalMode"] = "Rolling Updated"

#     # =========================
#     # Tentukan metrik utama yang akan digunakan
#     # =========================
#     if use_updated_metrics:
#         # Gunakan metrik Rolling Updated
#         primary_metrics = metrics_updated.copy()
#         primary_pred = pred_updated
#         print(f"✓ Menggunakan Rolling Updated metrics - MAPE: {metrics_updated.get('MAPE', 'N/A')}%")
#     else:
#         # Fallback ke OOS Forecast
#         primary_metrics = metrics_oos.copy()
#         primary_pred = pred_oos
#         print(f"✓ Menggunakan OOS Forecast metrics - MAPE: {metrics_oos.get('MAPE', 'N/A')}%")

#     # =========================
#     # 3️⃣ Forecast Masa Depan
#     # =========================
#     model_full = ARIMA(s, order=order)
#     fit_full = model_full.fit()

#     future = fit_full.get_forecast(steps=horizon).predicted_mean
#     future_index = pd.date_range(
#         start=s.index[-1] + pd.Timedelta(days=1),
#         periods=horizon,
#         freq="D"
#     )

#     future = pd.Series(future.values, index=future_index, name="forecast")

#     return {
#         "forecast": future,
#         "train": train,
#         "test": test,
#         "test_pred": primary_pred,  # CHANGED: gunakan prediksi dari metode yang dipilih
#         "test_pred_oos": pred_oos,
#         "test_pred_updated": pred_updated,
