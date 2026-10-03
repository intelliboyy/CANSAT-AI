"""Load raw weather observations and write a model-ready feature table."""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from cansat_ai.config import (
    DATETIME_COLUMN,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    SCALE_FEATURES,
    WEATHER_COLUMN,
)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Parse timestamps and add hour, day, month, and weekday columns."""
    frame = df.copy()
    frame[DATETIME_COLUMN] = pd.to_datetime(frame[DATETIME_COLUMN])
    frame["Hour"] = frame[DATETIME_COLUMN].dt.hour
    frame["Day"] = frame[DATETIME_COLUMN].dt.day
    frame["Month"] = frame[DATETIME_COLUMN].dt.month
    frame["Weekday"] = frame[DATETIME_COLUMN].dt.weekday
    return frame


def preprocess(raw_path=RAW_DATA_PATH, output_path=PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Normalize numeric sensors, keep calendar features, and encode weather labels."""
    df = add_time_features(pd.read_csv(raw_path))

    scaler = StandardScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(df[SCALE_FEATURES]),
        columns=[f"{col}_norm" for col in SCALE_FEATURES],
    )
    scaled["Hour"] = df["Hour"].to_numpy()
    scaled["Day"] = df["Day"].to_numpy()
    scaled["Month"] = df["Month"].to_numpy()
    scaled["Weekday"] = df["Weekday"].to_numpy()
    scaled["Weather_Label"] = LabelEncoder().fit_transform(df[WEATHER_COLUMN])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scaled.to_csv(output_path, index=False)
    return scaled


if __name__ == "__main__":
    out = preprocess()
    print(f"Wrote {len(out)} rows to {PROCESSED_DATA_PATH}")
