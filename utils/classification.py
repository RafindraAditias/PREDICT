def classify_pm25(value: float):
    """
    Kategori berdasarkan ambang PM2.5 (µg/m³) sesuai standar Indonesia.
    """
    v = float(value)
    if v <= 15.5:
        return "Baik", "green"
    if v <= 55.4:
        return "Sedang", "orange"
    if v <= 150.4:
        return "Tidak Sehat", "yellow"
    if v <= 250.4:
        return "Sangat Tidak Sehat", "red"
    return "Berbahaya", "darkred"


def trend_label(last_7_values):
    """
    Label tren sederhana berdasarkan slope linier.
    """
    vals = list(last_7_values)
    if len(vals) < 2:
        return "Stabil"

    x0 = vals[0]
    x1 = vals[-1]
    delta = x1 - x0

    if delta > 1.0:
        return "Meningkat"
    if delta < -1.0:
        return "Menurun"
    return "Stabil"


def volatility_label(last_14_values):
    """
    Label volatilitas berdasarkan koefisien variasi.
    """
    import numpy as np

    vals = np.array(list(last_14_values), dtype=float)
    if len(vals) < 2:
        return "Rendah"

    mean = float(np.mean(vals))
    std = float(np.std(vals))
    if mean == 0:
        return "Rendah"

    cv = std / mean
    if cv >= 0.35:
        return "Tinggi"
    if cv >= 0.20:
        return "Sedang"
    return "Rendah"