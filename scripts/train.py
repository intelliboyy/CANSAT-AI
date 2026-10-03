#!/usr/bin/env python3
"""Run preprocessing, weather classification, and temperature regression."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cansat_ai.classify_weather import train_classifier
from cansat_ai.predict_temperature import train_regressor
from cansat_ai.preprocess import preprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CanSat AI weather models.")
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Use the existing processed CSV instead of rebuilding it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_preprocess:
        processed = preprocess()
        print(f"[preprocess] {len(processed)} rows written")

    classification = train_classifier()
    print(f"[classify] accuracy = {classification['accuracy']:.4f}")

    regression = train_regressor()
    print(f"[regress] mse = {regression['mse']:.4f}  r2 = {regression['r2']:.4f}")


if __name__ == "__main__":
    main()
