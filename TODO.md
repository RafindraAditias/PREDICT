# TODO: Remove Hardcoded ARIMA Metrics

## ✅ Completed Steps

- [x] Analyze current implementation
- [x] Create plan
- [x] Get user approval

## ✅ Completed Steps

- [x] Analyze current implementation
- [x] Create plan
- [x] Get user approval
- [x] Update `models/arima_model.py` to add `use_updated_metrics` parameter
- [x] Update `app.py` to remove hardcoded ARIMA metrics

## 🔄 In Progress Steps

- [ ] Test the changes

## 📝 Detailed Steps

### Step 1: Update models/arima_model.py ✅

- [x] Add `use_updated_metrics=True` parameter to function signature
- [x] Add logic to select primary metrics (Updated vs OOS)
- [x] Update return dictionary to include `metrics` as primary
- [x] Keep `metrics_oos` and `metrics_updated` for reference

### Step 2: Update app.py ✅

- [x] Remove hardcoded MAPE, MAE, RMSE values in `extract_metrics()`
- [x] Update `run_model_cached()` to pass `use_updated_metrics` to ARIMA
- [x] Ensure ARIMA uses same structure as SARIMA

### Step 3: Testing

- [ ] Run the application
- [ ] Verify ARIMA metrics are calculated dynamically
- [ ] Check that values change based on actual data
