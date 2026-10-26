# Fixed P4 protocol — 2026-10-21

User authorization replaces old gates; historical Gate A was never approved. Only 2024/2025 public collisions and the official guide. This protocol is fixed before fitting or 2025 performance evaluation.

- Unit unique collision_index. Recorded collision_severity 1/2 -> positive, 3 -> negative. Exclude invalid targets with counts; duplicate IDs/year conflicts stop.
- Guide data_guide_2025.xlsx / 2024_code_list. Both files expose the same 44 fields. Current junction_detail, not historic. The conversion sheet has an apparent weather/road-surface labelling inconsistency; do not apply it. Current code-list definitions govern the already-converted data. November 2025 junction revision is noted on source page.
- A: speed_limit, road_type, junction_detail, first_road_class. B: A plus light_conditions, weather_conditions, road_surface_conditions, day_of_week, hour from HH:MM.
- Exclude ID/year/date/location/police force, casualty/vehicle counts, all severity/adjustment/injury-based fields, attendance, hazards and every non-allowlisted field.
- Speed numeric mph, valid {20,30,40,50,60,70}; -1/99/null -> missing. Other codes -> missing and count invalid. Categorical -1/null -> __MISSING__; unknown 9 (road/weather/surface),99 (junction) -> __UNKNOWN__. Light 7 remains 'Darkness - lighting unknown'. Other invalid categories -> missing/count. No dropping feature-missing rows. One-hot train-only, unseen categories ignored; numeric median imputation, indicator and standardization train-only.
- Sorted 2024 IDs split80/20 stratified seed20261021. All eligible 2024 participate in train/validation; no refit on validation after thresholds. Entire2025 final temporal test; never random-mixed. Pilot15000 solely from final training, internally80/20; no final-validation/test use.
- Majority: constant training prior probability, threshold.5, not F1-optimized.
- LR A/B: C1,lbfgs,max_iter2000,no class weights for natural probability distribution.
- RF B:150 trees,depth12,min_leaf20,sqrt features,bootstrap,no class weights,n_jobs1,seed.
- HGB B:learning_rate.05,max_iter150,max_leaf_nodes15,min_leaf20,l2=1,early_stoppingFalse,seed. Same one-hot representation, no ordering of nominal codes.
- No parameter search. Main validation-max-F1 threshold; ties fewerFP then higherthreshold, score ties grouped. Also .5 and validation-only grid .05:.05:.95. Freeze all choices before test. No post-test calibration or model changes.
- Metrics TP/FP/FN/TN,P/R/F1,FPR,AP(primary PR summary),trapezoidal PR-AUC(separate),ROC-AUC,Brier mean((p-y)^2),accuracy(supplement). Undefined precision NaN. Score every eligible test row.
- Explanation: LR B coefficients; original-column permutation AP decrease for LR B/RF B/HGB B, three repeats, shared fixed stratified validation sample<=5000. No test tuning. A/B joint increment is not a causal decomposition.
- Save predictions/splits/hashes/parameters/logs; rerun scoring/figures; repeat same-platform training if feasible. Single year-ahead test cannot prove stable multi-year improvement. No occurrence prediction, causal inference or production claim. Police labels/reporting may change.

Execution-only revision after the first reproducibility check: RF prediction with n_jobs=2 differed by <=1.67e-16 between runs, changing boundary decisions for a few rows. Original runs are retained as original-final/repeat-20261021. Use n_jobs=1 for deterministic probability accumulation; all statistical settings, data, selection rules and metrics unchanged. This is not test-performance tuning.
