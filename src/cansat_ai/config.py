"""Project paths and shared constants."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "Weather_Data.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "preprocessed_weather_data.csv"

DATETIME_COLUMN = "Date/Time"
WEATHER_COLUMN = "Weather"
TEMPERATURE_COLUMN = "Temp_C"

SCALE_FEATURES = [
    "Dew Point Temp_C",
    "Rel Hum_%",
    "Wind Speed_km/h",
    "Visibility_km",
    "Press_kPa",
]

TIME_FEATURES = ["Hour", "Day", "Month", "Weekday"]

REGRESSION_FEATURES = SCALE_FEATURES + TIME_FEATURES
