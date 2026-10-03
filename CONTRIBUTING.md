# Contributing

1. Fork the repository and create a feature branch.
2. Keep training scripts deterministic (`random_state=42`) unless you are intentionally exploring.
3. Place new raw files under `data/raw/` and regenerate processed data with `python scripts/train.py`.
4. Run `pytest -q` before opening a pull request.
5. Follow existing module names in `src/cansat_ai/` rather than adding one-off scripts at the repo root.
