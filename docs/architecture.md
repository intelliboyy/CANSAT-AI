# Architecture

This repository is a small machine-learning pipeline for CanSat-style weather analysis.

## Data flow

```
data/raw/Weather_Data.csv
        │
        ▼
src/cansat_ai/preprocess.py
        │
        ▼
data/processed/preprocessed_weather_data.csv
        │
        ├──► classify_weather.py     (Random Forest → weather label)
        └──► predict_temperature.py  (Linear Regression → Temp_C)
```

`scripts/train.py` runs the three stages in order and prints evaluation metrics.

## Models

| Task | Algorithm | Target | Features |
| --- | --- | --- | --- |
| Weather classification | Random Forest (100 trees) | Encoded `Weather` | Normalized dew point, humidity, wind, visibility, pressure, plus calendar fields |
| Temperature regression | Linear Regression | `Temp_C` | Same sensors unscaled then standardized, plus calendar fields |

Both models use an 80/20 train-test split with `random_state=42`.

## Dataset columns (raw)

- `Date/Time`
- `Temp_C`
- `Dew Point Temp_C`
- `Rel Hum_%`
- `Wind Speed_km/h`
- `Visibility_km`
- `Press_kPa`
- `Weather`
