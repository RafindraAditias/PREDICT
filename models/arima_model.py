import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from utils.metrics import calc_metrics

def run_arima_model(series: pd.Series, horizon: int = 30, test_size: int = 90, order=(1, 0, 1)):
    s = series.dropna().astype(float).sort_index()
    if len(s) <= test_size + 10:
        raise ValueError("Data terlalu sedikit untuk split train test. Kurangi test_size atau tambah data..")

    train = s.iloc[:-test_size]
    test = s.iloc[-test_size:]

    model = ARIMA(train, order=order)
    fit = model.fit()

    pred_test = fit.get_forecast(steps=len(test)).predicted_mean
    pred_test.index = test.index

    metrics = calc_metrics(test.values, pred_test.values)
    metrics["Model"] = f"ARIMA{order}"
    metrics["TestSize"] = int(test_size)

    model_full = ARIMA(s, order=order)
    fit_full = model_full.fit()

    future = fit_full.get_forecast(steps=horizon).predicted_mean
    future_index = pd.date_range(start=s.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    future = pd.Series(future.values, index=future_index, name="forecast")

    return {
        "forecast": future,
        "test_pred": pred_test,
        "train": train,
        "test": test,
        "metrics": metrics,
        "params": {"order": order}
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
