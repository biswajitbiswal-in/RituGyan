# Operational GFS Stage 1 Evaluation

Model artifact: `models\regime_classifier\stage1_lightgbm_5class_operational_gfs_v1`
Feature store version: `stage1_operational_gfs_v1`
Feature count: 122
Seed: 42; deterministic CPU training with one thread.

The model uses GFS-only operational predictors and unchanged Candidate A proxy labels. Western Disturbance is excluded from ML partitions.

## Split Metrics

### Train (243 samples)

| Metric | Value |
|---|---:|
| accuracy | 0.995885 |
| balanced_accuracy | 0.998675 |
| macro_precision | 0.993939 |
| macro_recall | 0.998675 |
| macro_f1 | 0.996259 |
| weighted_f1 | 0.995910 |

Confusion matrix (rows=true, columns=predicted):

```text
17 0 0 0 0
0 150 0 1 0
0 0 35 0 0
0 0 0 32 0
0 0 0 0 8
```

Per-class metrics:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| ACTIVE_MONSOON | 1.000000 | 1.000000 | 1.000000 | 17 |
| BREAK_MONSOON | 1.000000 | 0.993377 | 0.996678 | 151 |
| MONSOON_DEPRESSION | 1.000000 | 1.000000 | 1.000000 | 35 |
| OROGRAPHIC_MONSOON | 0.969697 | 1.000000 | 0.984615 | 32 |
| COASTAL_REGIME | 1.000000 | 1.000000 | 1.000000 | 8 |

### Validation (119 samples)

| Metric | Value |
|---|---:|
| accuracy | 0.781513 |
| balanced_accuracy | 0.492927 |
| macro_precision | 0.478941 |
| macro_recall | 0.492927 |
| macro_f1 | 0.470606 |
| weighted_f1 | 0.764103 |

Confusion matrix (rows=true, columns=predicted):

```text
1 2 3 0 2
0 75 4 2 1
0 2 12 1 0
0 0 3 5 0
2 3 1 0 0
```

Per-class metrics:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| ACTIVE_MONSOON | 0.333333 | 0.125000 | 0.181818 | 8 |
| BREAK_MONSOON | 0.914634 | 0.914634 | 0.914634 | 82 |
| MONSOON_DEPRESSION | 0.521739 | 0.800000 | 0.631579 | 15 |
| OROGRAPHIC_MONSOON | 0.625000 | 0.625000 | 0.625000 | 8 |
| COASTAL_REGIME | 0.000000 | 0.000000 | 0.000000 | 6 |

### Test (122 samples)

| Metric | Value |
|---|---:|
| accuracy | 0.811475 |
| balanced_accuracy | 0.661123 |
| macro_precision | 0.645045 |
| macro_recall | 0.661123 |
| macro_f1 | 0.649142 |
| weighted_f1 | 0.810986 |

Confusion matrix (rows=true, columns=predicted):

```text
3 1 0 2 1
1 66 4 1 1
2 6 19 0 0
0 0 3 10 0
0 1 0 0 1
```

Per-class metrics:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| ACTIVE_MONSOON | 0.500000 | 0.428571 | 0.461538 | 7 |
| BREAK_MONSOON | 0.891892 | 0.904110 | 0.897959 | 73 |
| MONSOON_DEPRESSION | 0.730769 | 0.703704 | 0.716981 | 27 |
| OROGRAPHIC_MONSOON | 0.769231 | 0.769231 | 0.769231 | 13 |
| COASTAL_REGIME | 0.333333 | 0.500000 | 0.400000 | 2 |

## Retrospective Baseline Comparison

| Split | Metric | Retrospective ERA5 baseline | Operational GFS | Difference |
|---|---|---:|---:|---:|
| validation | accuracy | 0.865546 | 0.781513 | -0.084034 |
| validation | balanced_accuracy | 0.675244 | 0.492927 | -0.182317 |
| validation | macro_precision | 0.596181 | 0.478941 | -0.117240 |
| validation | macro_recall | 0.675244 | 0.492927 | -0.182317 |
| validation | macro_f1 | 0.629956 | 0.470606 | -0.159350 |
| validation | weighted_f1 | 0.848416 | 0.764103 | -0.084313 |
| test | accuracy | 0.819672 | 0.811475 | -0.008197 |
| test | balanced_accuracy | 0.605079 | 0.661123 | +0.056044 |
| test | macro_precision | 0.599048 | 0.645045 | +0.045997 |
| test | macro_recall | 0.605079 | 0.661123 | +0.056044 |
| test | macro_f1 | 0.600471 | 0.649142 | +0.048671 |
| test | weighted_f1 | 0.816653 | 0.810986 | -0.005667 |

The comparison is descriptive: both models use the same ERA5-conditioned Candidate A labels; only predictor provenance changes. It does not establish independent meteorological truth.

Detailed machine-readable metrics are in `evaluation_metrics.json`; provenance is in `metadata.json`.
