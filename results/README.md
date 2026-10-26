# Actual P4 outputs

`data/` contains verified schema, sample counts, special-code missingness and dictionary extraction.

`final-20261021/` is authoritative. `metrics.csv` includes validation and test, main validation-F1 thresholds and default 0.5; majority uses its fixed policy. `average_precision` and `pr_auc` are distinct. Blank precision means no positive prediction. `test_predictions.csv.gz` and `validation_predictions.csv.gz` store labels and probabilities in deterministic cleaned/sorted order, without IDs. Every final test evaluation includes 101,525 rows. See `frozen_choices.json`, `run.json` and `training.txt` for actual settings and execution records.

`repeat-20261021/` is the full deterministic replication. `pilot-20261021/` uses only a subset of 2024 training and is not a temporal test.

`original-final-20261021/` and `original-repeat-20261021/` preserve initial RF n_jobs=2 runs. Floating-point accumulation differences up to 1.67e-16 caused threshold-boundary test counts to differ (2 TP, 8 FP). Final runs use n_jobs=1, with unchanged statistical settings and evaluation. Original evidence is not deleted or relabelled as final.

Contains public sector information licensed under OGL v3.0. Source: Department for Transport, STATS19. See `data/README.md` for source and licence links.

Final visual check corrected a display-only speed-limit category mismatch: 2024 float labels (20.0) and 2025 integer labels (20) are now normalized to the same mph category. This changes distribution aggregation labels and chart layout only, not model inputs or metrics. A regression test checks category alignment and yearly count totals. The initial figure is retained in original-final-20261021.
