# Tests

Run `.venv/bin/python -m unittest discover -s tests -v` from the project root. Eleven focused tests cover labels, years, predictors, missingness, duplicate IDs, train-only transformations, metric definitions, ties, reproducibility and the figure-table contract. End-to-end saved-prediction verification is in `scripts/verify.py`.
