## SAMPLES

| year | eligible_rows | fatal | serious | slight | severe_fraction |
|---|---|---|---|---|---|
| 2024 | 100927 | 1502 | 23567 | 75858 | 0.2484 |
| 2025 | 101525 | 1453 | 25191 | 74881 | 0.2624 |

## PERFORMANCE

| model | precision | recall | f1 | average_precision | roc_auc | brier |
|---|---|---|---|---|---|---|
| Majority | NA | 0.0000 | 0.0000 | 0.2624 | 0.5000 | 0.1938 |
| Logistic A | 0.2763 | 0.8676 | 0.4191 | 0.3195 | 0.5712 | 0.1911 |
| Logistic B | 0.2860 | 0.7970 | 0.4209 | 0.3299 | 0.5841 | 0.1901 |
| Random forest B | 0.2835 | 0.8450 | 0.4245 | 0.3363 | 0.5889 | 0.1897 |
| Gradient boosting B | 0.2818 | 0.8663 | 0.4253 | 0.3354 | 0.5887 | 0.1895 |

## COUNTS

| model | threshold | tp | fp | fn | tn | fpr |
|---|---|---|---|---|---|---|
| Majority | 0.5000 | 0 | 0 | 26644 | 74881 | 0.0000 |
| Logistic A | 0.2103 | 23116 | 60557 | 3528 | 14324 | 0.8087 |
| Logistic B | 0.2160 | 21235 | 53018 | 5409 | 21863 | 0.7080 |
| Random forest B | 0.2214 | 22515 | 56909 | 4129 | 17972 | 0.7600 |
| Gradient boosting B | 0.2080 | 23082 | 58821 | 3562 | 16060 | 0.7855 |

## INCREMENTS

| comparison | f1 | average_precision | roc_auc | brier |
|---|---|---|---|---|
| hist_gradient_B minus logistic_B | 0.0044 | 0.0056 | 0.0046 | -0.0007 |
| logistic_B minus logistic_A | 0.0018 | 0.0104 | 0.0129 | -0.0010 |
| random_forest_B minus logistic_B | 0.0036 | 0.0064 | 0.0048 | -0.0005 |
