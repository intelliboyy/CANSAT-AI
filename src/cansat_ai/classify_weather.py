"""Random Forest classifier for encoded weather conditions."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from cansat_ai.config import PROCESSED_DATA_PATH


def train_classifier(
    data_path=PROCESSED_DATA_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 100,
) -> dict:
    """Train on preprocessed features and return hold-out accuracy."""
    data = pd.read_csv(data_path)
    x = data.drop(columns=["Weather_Label"])
    y = data["Weather_Label"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    return {
        "model": model,
        "accuracy": accuracy,
        "report": classification_report(y_test, y_pred, zero_division=0),
    }


if __name__ == "__main__":
    result = train_classifier()
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(result["report"])
