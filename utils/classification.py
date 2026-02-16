def classify_pm25(value: float):
    """
    Kategori sederhana berbasis ambang PM2.5 (µg/m³).
    Kamu boleh sesuaikan threshold ini dengan standar yang kamu pakai di skripsi.
    """
    v = float(value)
    if v <= 15:
        return "Good", "green"
    if v <= 35:
        return "Moderate", "orange"
    if v <= 55:
        return "Unhealthy", "red"
    return "Hazardous", "darkred"


def trend_label(last_7_values):
    """
    Label tren sederhana berdasarkan slope linier ringan.
    """
    vals = list(last_7_values)
    if len(vals) < 2:
        return "Flat"

    x0 = vals[0]
    x1 = vals[-1]
    delta = x1 - x0

    if delta > 1.0:
        return "Increasing"
    if delta < -1.0:
        return "Decreasing"
    return "Flat"


def volatility_label(last_14_values):
    """
    Label volatilitas sederhana berdasarkan koefisien variasi.
    """
    import numpy as np

    vals = np.array(list(last_14_values), dtype=float)
    if len(vals) < 2:
        return "Low"

    mean = float(np.mean(vals))
    std = float(np.std(vals))
    if mean == 0:
        return "Low"

    cv = std / mean
    if cv >= 0.35:
        return "High"
    if cv >= 0.20:
        return "Medium"
    return "Low"
