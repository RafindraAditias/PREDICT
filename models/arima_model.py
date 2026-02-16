import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from utils.metrics import calc_metrics

def run_arima_model(series: pd.Series, horizon: int = 30, test_size: int = 90, order=(1, 0, 1)):
    s = series.dropna().astype(float).sort_index()
    if len(s) <= test_size + 10:
        raise ValueError("Data terlalu sedikit untuk split train test. Kurangi test_size atau tambah data.")

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
