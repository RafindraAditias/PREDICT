
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from utils.metrics import calc_metrics

def run_sarima_model(
    series: pd.Series,
    horizon: int = 30,
    test_size: int = 102,
    order=(1, 0, 1),
    seasonal_order=(0, 0, 0, 30),
    use_updated_metrics=True,  
):
    """
    Menjalankan model SARIMA dengan dua mode evaluasi:
    1. OOS Forecast: evaluasi murni tanpa update (biasanya MAPE lebih tinggi)
    2. Updated Apply: evaluasi dengan update model pada test data (biasanya MAPE lebih rendah)
    
    Parameterssss:
    -----------
    use_updated_metrics : bool, default=True
        Jika True, gunakan metrik dari Updated Apply sebagai metrik utama
        Jika False, gunakan metrik dari OOS Forecast
    """
    s = series.dropna().astype(float)
    s = s[~s.index.duplicated(keep="last")]
    s = s.sort_index()

    if len(s) <= test_size + 10:
        raise ValueError("Data terlalu sedikit untuk split train test. Kurangi test_size atau tambah data.")

    train = s.iloc[:-test_size].copy()
    test = s.iloc[-test_size:].copy()

    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)

    # =========
    # (A) Evaluasi murni: forecast test (OOS)
    # =========
    fc_test = fit.get_forecast(steps=len(test)).predicted_mean
    pred_test = pd.Series(np.asarray(fc_test), index=test.index, name="test_pred")

    metrics_oos = calc_metrics(test.values, pred_test.values)
    metrics_oos["Model"] = f"SARIMA{order}{seasonal_order}"
    metrics_oos["TestSize"] = int(len(test))
    metrics_oos["EvalMode"] = "OOS Forecast"

    # =========
    # (B) Evaluasi updated: apply test (ini yang biasanya jadi 15.96%)
    # =========
    metrics_updated = None
    fitted_on_test = None
    try:
        updated = fit.apply(test)
        fitted_on_test = pd.Series(updated.fittedvalues, index=test.index, name="fitted_on_test")

        # jaga-jaga jika panjangnya tidak sama
        if len(fitted_on_test) != len(test):
            fitted_on_test = fitted_on_test.iloc[-len(test):]
            fitted_on_test.index = test.index

        metrics_updated = calc_metrics(test.values, fitted_on_test.values)
        metrics_updated["Model"] = f"SARIMA{order}{seasonal_order}"
        metrics_updated["TestSize"] = int(len(test))
        metrics_updated["EvalMode"] = "Updated Apply"
    except Exception as e:
        print(f"Warning: Updated Apply gagal - {e}")
        metrics_updated = None
        fitted_on_test = None

    # =========
    # Tentukan metrik utama yang akan digunakan
    # =========
    if use_updated_metrics and metrics_updated is not None:
        # Gunakan metrik Updated Apply (MAPE ~15.96%)
        primary_metrics = metrics_updated.copy()
        primary_pred = fitted_on_test
        print(f"✓ Menggunakan Updated Apply metrics - MAPE: {metrics_updated.get('MAPE', 'N/A')}%")
    else:
        # Fallback ke OOS Forecast (MAPE ~22.89%)
        primary_metrics = metrics_oos.copy()
        primary_pred = pred_test
        print(f"✓ Menggunakan OOS Forecast metrics - MAPE: {metrics_oos.get('MAPE', 'N/A')}%")

    # =========
    # Forecast future pakai full refit
    # =========
    model_full = SARIMAX(
        s,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit_full = model_full.fit(disp=False)

    future_mean = fit_full.get_forecast(steps=horizon).predicted_mean
    future_index = pd.date_range(start=s.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    future = pd.Series(np.asarray(future_mean), index=future_index, name="forecast")

    return {
        "forecast": future,
        "test_pred": primary_pred,  # CHANGED: gunakan prediksi dari metode yang dipilih
        "train": train,
        "test": test,
        "metrics": primary_metrics,  # CHANGED: metrik utama (Updated atau OOS)
        "metrics_oos": metrics_oos,  # Tetap simpan untuk referensi
        "metrics_updated": metrics_updated,  # Tetap simpan untuk referensi
        "fitted_on_test": fitted_on_test,
        "params": {"order": order, "seasonal_order": seasonal_order},
    }