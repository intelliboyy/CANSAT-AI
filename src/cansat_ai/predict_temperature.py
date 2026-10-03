"""Linear regression model for air temperature from related sensors."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from cansat_ai.config import RAW_DATA_PATH, REGRESSION_FEATURES, TEMPERATURE_COLUMN
from cansat_ai.preprocess import add_time_features


def train_regressor(
    data_path=RAW_DATA_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Fit a linear model and return MSE and R² on the test split."""
    df = add_time_features(pd.read_csv(data_path))
    x_scaled = StandardScaler().fit_transform(df[REGRESSION_FEATURES])
    y = df[TEMPERATURE_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled, y, test_size=test_size, random_state=random_state
    )
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return {
        "model": model,
        "mse": mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }


if __name__ == "__main__":
    result = train_regressor()
    print(f"Mean Squared Error: {result['mse']:.4f}")
    print(f"R2 Score: {result['r2']:.4f}")
