# Anomaly detector evaluation

Dataset: 50,000 synthetic trades per seed, ground truth from the generator's
injection plan. Regenerate with `python scripts/evaluate_detector.py`.

## Threshold sweep for UNUSUAL_QUANTITY (tuning seed only)

Raw-scale vs log-scale MAD. The raw-scale rule cannot be fixed by moving k:
it is flagging the lognormal's legitimate right tail at every threshold.

| space | k | flags | precision | recall | F1 |
|---|---|---|---|---|---|
| raw | 5.0 | 2267 | 0.034 | 1.000 | 0.067 |
| raw | 10.0 | 572 | 0.136 | 1.000 | 0.240 |
| log | 3.0 | 133 | 0.541 | 0.923 | 0.682 |
| log | 3.25 | 91 | 0.747 | 0.872 | 0.805 |
| log | 3.5 | 73 | 0.904 | 0.846 | 0.874 |
| log | 3.75 | 64 | 0.969 | 0.795 | 0.873 |
| log | 4.0 | 57 | 0.982 | 0.718 | 0.830 |
| log | 4.5 | 43 | 0.977 | 0.538 | 0.694 |

Selected **k = 3.5** (log space) by F1 on seed 42.

## Held-out evaluation (threshold frozen)

Seeds 7, 13, 101, 2024, 31337 - never used for tuning.

| rule | precision | recall | F1 |
|---|---|---|---|
| HIGH_LATENCY | 1.000 | 0.997 | 0.999 |
| UNUSUAL_QUANTITY | 0.909 | 0.864 | 0.886 |
| HIGH_FAILURE_RATE | 0.920 | 1.000 | 0.956 |

Means over the five held-out seeds. HIGH_FAILURE_RATE scores on a handful of
accounts per seed, so its figures are coarse-grained by construction.
