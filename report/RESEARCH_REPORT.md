# Predicting Road Collision Severity Using Explainable Machine Learning: A Temporal Evaluation of Great Britain's STATS19 Data

**Research-training computational study for Dezhang Li's Computer Science portfolio**  
Completed locally: 21 October 2026. Not peer reviewed. See the contribution disclosure below.

## Abstract

This study evaluates whether recorded road, environmental and time attributes discriminate fatal or serious collisions from slight collisions in Great Britain. Official STATS19 collision tables provide 100,927 eligible records for 2024 and 101,525 for 2025. A stratified 2024 training/validation split supports preprocessing and threshold selection; all 2025 records form a subsequent-year test. Five configurations compare a majority baseline, road-only logistic regression, expanded logistic regression, random forest and histogram gradient boosting. Adding environmental and temporal variables increases test Average Precision from 0.3195 to 0.3299. Random forest obtains the highest Average Precision, 0.3363, while boosting has the highest F1, 0.4253. These are modest descriptive improvements. Thresholds maximizing validation F1 yield recall around 0.80–0.87 but false-positive rates around 0.71–0.81. Speed limit and road type dominate validation permutation importance. The findings concern retrospective classification of already-recorded injury collisions, not collision occurrence, causal road-safety effects or operational deployment. Saved predictions, source hashes, tests and deterministic retraining support reproducibility.

## 1. Introduction

Road-collision records offer an accessible setting for learning applied machine learning, but the prediction target requires careful definition. Classifying the severity of an already-recorded collision differs from predicting whether a collision will happen. The latter would require information about journeys or road exposure without collisions, which this dataset does not contain. Here, the outcome is the recorded binary severity category conditional on a collision appearing in STATS19.

The main question is whether broader contextual information and more flexible models provide useful discrimination beyond basic road attributes. RQ1 asks how well road type, speed limit, junction detail and road class distinguish fatal/serious from slight collisions. RQ2 asks whether lighting, weather, surface condition, weekday and hour add predictive information. RQ3 asks whether random forest and gradient boosting improve upon expanded logistic regression in the following year. “Stable improvement” cannot be established from one year transition; the design supplies one temporal comparison.

The contribution is a reproducible research exercise rather than a new algorithm. It emphasizes official data definitions, explicit feature exclusions, natural class proportions and evaluation that includes both discrimination and probability error. A modest or negative comparison remains informative. In particular, a high recall obtained by classifying most records as severe should not be mistaken for strong discrimination.

## 2. Related Work

Mannering and Bhat (2014) review methodological challenges in crash analysis, including unobserved heterogeneity, endogeneity and spatial or temporal dependence [3]. Their discussion motivates caution about interpreting associations or assuming independent observations. This project does not resolve those identification problems; it instead limits its claims to held-out predictive performance.

Umer et al. (2020) compare tree ensembles and a voting classifier on US accident records, using a random 70/30 split and a feature-importance analysis [4, Sections 3–4]. That work provides an example of model comparison in collision-related data. Its source population, severity coding and predictor set differ from the present task, so its reported scores are not benchmarks for STATS19. The present experiment also avoids selecting features using the complete dataset.

Random forests combine randomized trees, with feature sampling and bootstrap variation providing diversity [5]. Their flexibility makes them a reasonable comparison to a linear log-odds model, but does not guarantee improvement on these particular predictors. Histogram gradient boosting is included as a second nonlinear method with a fixed, modest configuration.

Metric definitions also matter. Average Precision is a recall-weighted summary, whereas trapezoidal PR-AUC linearly interpolates curve points [6]. They can differ substantially for tied scores. Probability evaluation complements thresholded classification: Brier loss combines aspects of reliability, resolution and outcome uncertainty, so a lower value alone does not prove better calibration [7].

## 3. Data and Methods

### 3.1 Sources, unit and recorded outcome

The Department for Transport publishes public, non-sensitive collision records for police-recorded personal-injury collisions on public roads in Great Britain [1]. Only the individual 2024 and 2025 collision CSVs and the official spreadsheet guide were acquired, under the Open Government Licence v3.0. No vehicle, casualty or sensitive-factor table was used.

The files contain 44 identically named fields. Their sizes are 19,208,488 and 19,349,153 bytes; the guide is 85,112 bytes. Download URLs, acquisition time and SHA-256 values are retained in the data manifest. The guide's current code list defines severity as Fatal=1, Serious=2 and Slight=3. The positive target combines codes 1 and 2; code 3 is negative. This uses recorded collision labels, not adjusted severity estimates or individual casualty outcomes.

Table 1 gives the complete eligible populations. No duplicate collision identifiers, invalid target codes, cross-year identifier overlap or date/year/weekday inconsistencies were found. Every row remained eligible.

<!-- SAMPLES:START -->
| year | eligible_rows | fatal | serious | slight | severe_fraction |
|---|---|---|---|---|---|
| 2024 | 100927 | 1502 | 23567 | 75858 | 0.2484 |
| 2025 | 101525 | 1453 | 25191 | 74881 | 0.2624 |
<!-- SAMPLES:END -->

### 3.2 Cleaning and cross-year compatibility

The current code list, rather than numeric intuition, determines valid categories. Missing codes and explicit Unknown codes are counted separately. Junction detail has 7,005 missing and 2 Unknown entries in 2024, versus 6,701 and 1,427 in 2025. Road-type Unknown counts are 2,568 and 1,953. Weather Unknown counts are 3,145 and 2,385. Road-surface missing/Unknown counts are 704/1,670 and 634/1,342. These differences remain visible rather than being removed through complete-case analysis.

Speed limit has three missing observations in 2024 and none in 2025; all observed valid limits are 20–70 mph in the declared categories. Hour is derived from valid HH:MM strings. No invalid selected codes or malformed times occurred. Lighting code 7 retains its partially informative meaning, darkness with unknown lighting. Full counts, including the six missing road-class values in 2025, are supplied in the missingness CSV.

The guide's legacy conversion sheet contains an apparent weather/road-surface label inconsistency. No legacy conversion is applied: both downloaded tables already expose current fields, and the current code list supplies their meanings. This avoids guessing a correction. Matching schemas do not prove identical recording practice. DfT describes revisions and injury-reporting changes [1,2], and the increased junction Unknown frequency remains a comparability limitation.

### 3.3 Predictors and leakage controls

Feature Set A contains speed_limit, road_type, junction_detail and first_road_class. Set B adds light_conditions, weather_conditions, road_surface_conditions, day_of_week and hour. These are attributes recorded for the collision; their use does not establish availability before an event.

The explicit allowlist excludes collision severity, all severity-adjustment and injury-based fields, casualty and vehicle counts, identifiers, exact location, police force, attendance and every other column. The year is used solely to define the temporal split. A unique identifier is used for ordering and integrity checks, never as a predictor. Excluding direct outcomes reduces obvious leakage, although reporting-related correlations may remain.

Nominal variables are one-hot encoded. Missing and Unknown categories receive explicit tokens; categories absent from training are ignored at transformation time. Speed is median-imputed with a missingness indicator and standardized. All learned transformations fit only on the training partition. Identical preprocessing conventions apply to both linear and tree models; no target encoding, oversampling or class reweighting is used.

### 3.4 Training, validation and models

Sorted 2024 records are stratified into 80,741 training and 20,186 validation observations, using seed 20261021. Their severe counts are 20,055 and 5,014. A preliminary 15,000-record subset of the training partition supports a 12,000/3,000 internal pilot. It exercises cleaning, fitting, evaluation and plotting without using final validation or 2025 performance.

The formal models fit the full 80% training partition. The remaining 20% selects thresholds; it is deliberately not folded back into fitting afterward. Thus all eligible 2024 observations participate in development, but not all are fitting observations. The complete 2025 population is reserved for final scoring.

The majority classifier always predicts Slight and assigns the training severe prevalence as a constant probability. Both logistic regressions use C=1, lbfgs and at most 2,000 iterations. Random forest uses 150 trees, maximum depth 12, minimum leaf size 20 and square-root feature sampling. Boosting uses 150 iterations, learning rate 0.05, at most 15 leaves, minimum leaf size 20 and L2 regularization 1, with early stopping disabled. There is no parameter search.

For each learned model, the main threshold maximizes 2024 validation F1, grouping equal scores and resolving ties by fewer false positives, then a higher threshold. The majority threshold stays at 0.5. All choices are saved before the runner loads 2025 model inputs. Additional tables retain the default 0.5 threshold and a validation-only threshold grid. No test-driven redesign or recalibration occurs.

### 3.5 Evaluation, explanation and reproducibility

For the positive severe class, Precision=TP/(TP+FP), Recall=TP/(TP+FN), F1=2TP/(2TP+FP+FN), and FPR=FP/(FP+TN). Precision is NA when no positive predictions exist. Each test configuration includes all 101,525 collisions. Average Precision is the primary PR summary; separately named trapezoidal PR-AUC, ROC-AUC and mean squared probability error (Brier score) are also saved.

Reliability plots use ten fixed-width probability bins, retaining bin counts. Permutation importance measures Average Precision decrease after shuffling each original feature on a shared stratified 5,000-record validation subset, with three repeats. Error bars represent variation across permutations, not inferential confidence intervals [8]. Logistic coefficients are saved; all category levels are retained under regularization, so individual coefficients are not reference-category odds ratios.

Twelve tests cover mapping, temporal separation, feature exclusions, missing/unseen categories, train-only transformations, duplicate detection, deterministic splitting, threshold ties, metrics and figure inputs. Independent rescoring checks every saved confusion count and metric. Full retraining verifies predictions and explanations. An execution correction, described below, removes random-forest parallel accumulation variability without changing statistical settings.

## 4. Results

### 4.1 Temporal performance

Table 2 reports 2025 performance at thresholds chosen using 2024 validation. NA for majority precision is intentional. All values are generated from the saved metric CSV.

<!-- PERFORMANCE:START -->
| model | precision | recall | f1 | average_precision | roc_auc | brier |
|---|---|---|---|---|---|---|
| Majority | NA | 0.0000 | 0.0000 | 0.2624 | 0.5000 | 0.1938 |
| Logistic A | 0.2763 | 0.8676 | 0.4191 | 0.3195 | 0.5712 | 0.1911 |
| Logistic B | 0.2860 | 0.7970 | 0.4209 | 0.3299 | 0.5841 | 0.1901 |
| Random forest B | 0.2835 | 0.8450 | 0.4245 | 0.3363 | 0.5889 | 0.1897 |
| Gradient boosting B | 0.2818 | 0.8663 | 0.4253 | 0.3354 | 0.5887 | 0.1895 |
<!-- PERFORMANCE:END -->

The majority classifier attains 0.7376 accuracy while missing every severe collision. This illustrates why accuracy is insufficient. Road-only logistic regression has ROC-AUC 0.5712 and AP 0.3195, exceeding constant ranking but providing limited separation. Expanded logistic regression reaches ROC-AUC 0.5841 and AP 0.3299.

Random forest has the highest AP and ROC-AUC, while boosting has the highest F1 and lowest Brier loss. These rankings do not support an unqualified “best model” claim. Table 2b shows the operational cost of validation-F1 thresholds: all learned classifiers label large portions of slight collisions as severe.

<!-- COUNTS:START -->
| model | threshold | tp | fp | fn | tn | fpr |
|---|---|---|---|---|---|---|
| Majority | 0.5000 | 0 | 0 | 26644 | 74881 | 0.0000 |
| Logistic A | 0.2103 | 23116 | 60557 | 3528 | 14324 | 0.8087 |
| Logistic B | 0.2160 | 21235 | 53018 | 5409 | 21863 | 0.7080 |
| Random forest B | 0.2214 | 22515 | 56909 | 4129 | 17972 | 0.7600 |
| Gradient boosting B | 0.2080 | 23082 | 58821 | 3562 | 16060 | 0.7855 |
<!-- COUNTS:END -->

Expanded logistic regression correctly identifies 21,235 severe collisions but also flags 53,018 slight collisions. Boosting identifies 23,082 severe collisions with 58,821 false positives. At the default 0.5 threshold, road-only logistic regression and random forest predict no positives; expanded logistic regression predicts ten and boosting three. Their high main-threshold recall therefore reflects a deliberate recall/precision trade-off, not near-certain severe predictions.

### 4.2 Incremental value and model complexity

Table 3 gives absolute metric changes. Positive AP, F1 and ROC-AUC changes are favorable; negative Brier changes are favorable.

<!-- INCREMENTS:START -->
| comparison | f1 | average_precision | roc_auc | brier |
|---|---|---|---|---|
| hist_gradient_B minus logistic_B | 0.0044 | 0.0056 | 0.0046 | -0.0007 |
| logistic_B minus logistic_A | 0.0018 | 0.0104 | 0.0129 | -0.0010 |
| random_forest_B minus logistic_B | 0.0036 | 0.0064 | 0.0048 | -0.0005 |
<!-- INCREMENTS:END -->

Adding the five environmental/time variables raises AP by 0.0104 and ROC-AUC by 0.0129, but F1 by only 0.0018. Recall decreases at the independently selected thresholds, while precision increases. The group comparison supports modest joint incremental value, not a claim that every added variable independently helps.

Compared with expanded logistic regression, random forest adds about 0.0064 AP and boosting about 0.0056. Their F1 improvements are approximately 0.0036 and 0.0044. On 2024 validation, however, both tree methods have slightly lower F1 than expanded logistic regression. Consequently, one temporal test suggests small improvements in some metrics, not stable superiority or statistical significance.

### 4.3 Probability behavior and explanations

The recorded severe fraction increases from 24.8387% in 2024 to 26.2438% in 2025. All models have lower ROC-AUC on the temporal test than on validation, despite higher test AP. These metrics respond differently to prevalence and ranking; higher AP across years cannot alone establish improved generalization.

Brier loss declines from 0.1938 for the constant baseline to 0.1895 for boosting. Reliability bins show underprediction in substantial portions of 2025. For expanded logistic regression, the 0.1–0.2 bin contains 17,390 observations, with mean probability 0.1649 and observed severe fraction 0.2015. The highest occupied bins contain only ten observations for logistic regression and three for boosting, so their visually large deviations should not receive equal weight.

Speed limit and road type rank first and second in permutation AP decrease for all three expanded models. For expanded logistic regression their mean decreases are approximately 0.0233 and 0.0205; hour ranks next. Secondary rankings differ across models. The standardized speed coefficient is positive, about 0.2062, but correlated road attributes and recording processes prevent a causal interpretation. Missing-category coefficients also show that models can exploit patterns of recording completeness.

### 4.4 Verification outcome

Final deterministic runs produce exactly equal stored predictions, metric tables, permutation values and logistic coefficients. Independent rescoring and all twelve tests pass. Figures are generated from CSV inputs whose hashes are recorded.

Initial random-forest parallel prediction differed between repetitions by at most approximately 1.67×10⁻¹⁶. Because some scores lay exactly at the threshold, the original test counts differed by two true positives and eight false positives. Switching only execution to one thread removed this variability. Original runs and logs remain available; the final tables use deterministic reruns. No outcome was edited to improve a score.

## 5. Discussion

RQ1 is answered by limited discrimination using road context. The road-only model ranks records somewhat better than a constant score, but at its F1-selected threshold its precision is only about 0.28. RQ2 receives a qualified positive answer: environment/time variables add information jointly, although the improvement is small and does not increase every thresholded metric. RQ3 receives an equally qualified answer: nonlinear models improve several 2025 metrics slightly, with rankings dependent on the metric and no evidence of stable advantage across multiple future years.

The large false-positive burden is central. F1 ignores true negatives, and optimizing it under weak class separation can favor predicting positive for most observations. The majority baseline provides a useful accuracy reference but is not a competitive severe-detection policy. Neither high majority accuracy nor high optimized recall demonstrates deployment readiness. No cost model, response workflow or acceptable false-alarm budget has been established.

These results are conditional on recorded injury collisions. They contain no denominator for ordinary travel and cannot estimate collision incidence. Combining Fatal and Serious also hides differences between those outcomes. Police recording and reporting-system changes can alter labels; the observed year difference is not evidence that road conditions caused injury severity to change. Adjusted official casualty statistics address a different estimation purpose and were not substituted into row-level labels.

Temporal separation prevents random mixing of years but does not remove recurring locations, shared police practices or spatial dependence. A random internal split may be optimistic for unfamiliar locations. Additional years and geographically separated evaluation would be useful future work, but were not performed. The present study makes no external-population, policy-intervention or real-time safety claim.

Interpretability is similarly bounded. Permuting correlated features can create unusual combinations or redistribute importance among related variables. Three repeats do not provide uncertainty about an entire population. Regularized coefficients and importance rankings describe fitted models, not traffic mechanisms. The joint feature increment also cannot isolate the causal or independent value of weather, lighting or time.

## 6. Conclusion

A complete temporal experiment using public STATS19 records shows modest incremental value from environmental and temporal attributes and small metric-dependent gains from tree ensembles. The strongest AP is 0.3363 and the strongest F1 is 0.4253; high recall comes with substantial false-positive rates. The principal contribution is a transparent computational workflow and an appropriately limited interpretation of its actual results. Broader generalization and practical utility remain untested.

## Contribution and AI assistance disclosure

Codex/AI implemented acquisition, preprocessing, models, tests, visualizations and the initial report under user direction in this session. This document does not establish that Dezhang Li independently wrote the code, conducted the analysis or approved the interpretation. Student review, understanding and any personal contribution statement must reflect actual participation. This is an unpublished research-training project, not a peer-reviewed paper.

## References

1. Department for Transport. **Road safety open data**, current annual CSVs and official data guide. Accessed 21 October 2026. https://www.gov.uk/government/statistical-data-sets/road-safety-open-data
2. Department for Transport. **Guide to injury-based reporting and severity adjustments for road casualty statistics**. https://www.gov.uk/government/publications/guide-to-severity-adjustments-for-reported-road-casualty-statistics
3. Mannering, F. L., & Bhat, C. R. (2014). **Analytic methods in accident research: Methodological frontier and future directions**. *Analytic Methods in Accident Research, 1*, 1–22. https://doi.org/10.1016/j.amar.2013.09.001 — [author-hosted full text](https://engineering.purdue.edu/~flm/CE615_files/Mannering-Bhat-AMAR-2014.pdf).
4. Umer, M., Sadiq, S., Ishaq, A., Ullah, S., Saher, N., & Madni, H. A. (2020). **Comparison Analysis of Tree Based and Ensembled Regression Algorithms for Traffic Accident Severity Prediction**. arXiv preprint. https://arxiv.org/abs/2010.14921
5. Breiman, L. (2001). **Random Forests**. *Machine Learning, 45*, 5–32. https://doi.org/10.1023/A:1010933404324
6. scikit-learn developers. **Average Precision and model evaluation**. https://github.com/scikit-learn/scikit-learn/blob/main/doc/modules/model_evaluation.rst
7. scikit-learn developers. **Probability calibration**. https://scikit-learn.org/stable/modules/calibration.html
8. scikit-learn developers. **Permutation feature importance**. https://scikit-learn.org/stable/modules/permutation_importance.html

Contains public sector information licensed under the [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/). Source: Department for Transport, STATS19.

## Figures and machine-readable evidence

[Workflow](../figures/final-20261021/workflow.png), [distributions](../figures/final-20261021/distributions.png), [model performance](../figures/final-20261021/model_performance.png), [PR curves](../figures/final-20261021/precision_recall.png), [importance](../figures/final-20261021/feature_importance.png), [calibration](../figures/final-20261021/calibration.png). All exact values, both threshold policies, curve points and bin counts are in [final results](../results/final-20261021/).
